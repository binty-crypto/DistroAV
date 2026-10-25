/******************************************************************************
	Copyright (C) 2016-2024 DistroAV <contact@distroav.org>

	This program is free software; you can redistribute it and/or
	modify it under the terms of the GNU General Public License
	as published by the Free Software Foundation; either version 2
	of the License, or (at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, see <https://www.gnu.org/licenses/>.
******************************************************************************/

#include "plugin-main.h"
#include "sync-debug.h"
#include <util/platform.h>
#include <util/threading.h>
#include <media-io/video-frame.h>

#include <QDesktopServices>
#include <QUrl>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#define TEXFORMAT GS_BGRA
#define FLT_PROP_NAME "ndi_filter_ndiname"
#define FLT_PROP_GROUPS "ndi_filter_ndigroups"
#define FLT_PROP_FPS_LIMIT "ndi_filter_fps_limit"
#define FLT_PROP_READBACK_DELAY "ndi_filter_readback_delay"
#define FLT_PROP_KEEP_ACTIVE "ndi_filter_keep_active"

// Asynchronous GPU readback
// -------------------------
// The stock filter stages the rendered texture and maps it in the same frame, which makes the
// render thread wait for the GPU copy to finish, once per filter per frame. With many
// "Dedicated NDI Output" filters that wait dominates the frame time.
//
// With readback_delay > 0 we keep a small ring of staging surfaces: each frame we stage into a
// free slot and map a slot that was staged `readback_delay` filter-frames ago, whose copy has
// long since finished, so map() returns immediately. The mapped pointer is handed to a per-filter
// worker thread that performs the NDI send; the render thread unmaps the slot on a later frame
// once the worker is done. The row-copy into video_output is skipped on this path.
#define MAX_STAGE_SLOTS 4
#define MAX_READBACK_DELAY (MAX_STAGE_SLOTS - 2)

enum stage_state { STAGE_FREE = 0, STAGE_STAGED, STAGE_MAPPED };

struct stage_slot {
	gs_stagesurf_t *surf;
	stage_state state;
	uint64_t seq;       // filter-frame number at which it was staged
	uint64_t timestamp; // OBS timestamp of the staged frame
	uint8_t *data;      // valid while STAGE_MAPPED
	uint32_t linesize;
	bool worker_done; // worker has finished with the mapped data (guarded by worker mutex)
};

struct readback_job {
	int slot;
	uint8_t *data;
	uint32_t linesize;
	uint64_t timestamp;
};

struct readback_worker {
	std::thread thread;
	std::mutex mutex;
	std::condition_variable cv;
	std::deque<readback_job> queue;
	bool stop = false;
	int in_flight = 0;
};

typedef struct {
	obs_source_t *obs_source;

	NDIlib_send_instance_t ndi_sender;

	pthread_mutex_t ndi_sender_video_mutex;
	pthread_mutex_t ndi_sender_audio_mutex;

	obs_video_info ovi;
	obs_audio_info oai;

	uint32_t known_width;
	uint32_t known_height;
	bool rendered;

	gs_texrender_t *texrender;
	gs_stagesurf_t *stagesurface;
	uint8_t *video_data;
	uint32_t video_linesize;

	// Frame rate limit (0 = match OBS)
	uint32_t fps_limit;
	uint64_t next_frame_ns;

	// Async readback ring
	uint32_t readback_delay; // 0 = synchronous (stock behaviour)
	uint32_t ring_size;      // readback_delay + 2 while async, 0 while sync
	stage_slot stage_ring[MAX_STAGE_SLOTS];
	uint64_t frame_seq;
	readback_worker *worker;
	bool dropped_logged;

	// Keep the parent source active while this filter is attached, so media and
	// slideshows play and the output is valid without the scene being in Program,
	// Preview or a Multiview.
	bool keep_active;          // setting
	bool holding_active;       // we currently hold an active ref on the parent
	obs_source_t *held_parent; // the parent we took the ref on
#ifdef SYNC_DEBUG
	// Time offset to apply to OBS timestamps to synchronize with NDI timestamps
	uint64_t obs_to_ndi_time_offset;
#endif
	video_t *video_output;
	bool is_audioonly;

	uint8_t *audio_conv_buffer;
	size_t audio_conv_buffer_size;
	int32_t no_video_connections;
	std::chrono::time_point<std::chrono::steady_clock> last_video_conn_check;
	int32_t no_audio_connections;
	std::chrono::time_point<std::chrono::steady_clock> last_audio_conn_check;
} ndi_filter_t;

const char *ndi_filter_getname(void *)
{
	return obs_module_text("NDIPlugin.FilterName");
}

const char *ndi_audiofilter_getname(void *)
{
	return obs_module_text("NDIPlugin.AudioFilterName");
}

#ifdef SYNC_DEBUG
uint64_t now_ns()
{
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					     std::chrono::system_clock::now().time_since_epoch())
					     .count());
}
#endif

void ndi_filter_update(void *data, obs_data_t *settings);
void ndi_sender_destroy(ndi_filter_t *filter);
void ndi_sender_create(ndi_filter_t *filter, obs_data_t *settings);

// Callback fired when the parent source is renamed. Recreate the NDI sender to pick up the new name.
void on_renamed(void *data, calldata_t *)
{
	auto f = (ndi_filter_t *)data;
	if (!f)
		return;
	obs_log(LOG_DEBUG, "on_parent_renamed: parent source renamed; recreating NDI sender for filter '%s'",
		obs_source_get_name(f->obs_source));
	// Recreate sender using current settings
	ndi_sender_create(f, nullptr);
}

// ---- keep parent active ---------------------------------------------------
// Holding an 'active' reference on the parent makes OBS treat it (and its
// children) as if it were in Program: media sources and slideshows play, and
// obs_source_showing() is true so the output is valid. Reference counted, so it
// coexists with the scene actually being in Program or Preview.

static void ndi_filter_release_active(ndi_filter_t *f)
{
	if (!f->holding_active)
		return;
	if (f->held_parent)
		obs_source_dec_active(f->held_parent);
	f->holding_active = false;
	f->held_parent = nullptr;
}

static void ndi_filter_apply_keep_active(ndi_filter_t *f)
{
	if (f->is_audioonly)
		return;
	obs_source_t *parent = obs_filter_get_parent(f->obs_source);
	bool want = f->keep_active && parent != nullptr;
	if (f->holding_active && (!want || f->held_parent != parent))
		ndi_filter_release_active(f);
	if (want && !f->holding_active) {
		obs_source_inc_active(parent);
		f->held_parent = parent;
		f->holding_active = true;
	}
}

static void ndi_filter_disconnect_rename_handlers(ndi_filter_t *filter)
{
	if (!filter || !filter->obs_source) {
		return;
	}

	obs_source_t *parent = obs_filter_get_parent(filter->obs_source);
	if (parent) {
		signal_handler_t *parent_sh = obs_source_get_signal_handler(parent);
		if (parent_sh) {
			signal_handler_disconnect(parent_sh, "rename", on_renamed, filter);
		}
	}

	signal_handler_t *filter_sh = obs_source_get_signal_handler(filter->obs_source);
	if (filter_sh) {
		signal_handler_disconnect(filter_sh, "rename", on_renamed, filter);
	}
}

obs_properties_t *ndi_filter_getproperties(void *priv)
{
	obs_log(LOG_DEBUG, "+ndi_filter_getproperties(...)");
	obs_properties_t *props = obs_properties_create();
	obs_properties_set_flags(props, OBS_PROPERTIES_DEFER_UPDATE);

	obs_property_t *ndi_name_property = obs_properties_add_text(
		props, FLT_PROP_NAME, obs_module_text("NDIPlugin.FilterProps.NDIName"), OBS_TEXT_DEFAULT);
	obs_property_set_long_description(ndi_name_property,
					  obs_module_text("NDIPlugin.FilterProps.NDIName.Description"));

	obs_properties_add_text(props, FLT_PROP_GROUPS, obs_module_text("NDIPlugin.FilterProps.NDIGroups"),
				OBS_TEXT_DEFAULT);

	obs_property_t *fps_property = obs_properties_add_list(props, FLT_PROP_FPS_LIMIT,
							       obs_module_text("NDIPlugin.FilterProps.FpsLimit"),
							       OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(fps_property, obs_module_text("NDIPlugin.FilterProps.FpsLimit.MatchOBS"), 0);
	const int fps_options[] = {60, 50, 30, 25, 15, 10, 5, 2, 1};
	for (int fps : fps_options) {
		char label[16];
		snprintf(label, sizeof(label), "%d fps", fps);
		obs_property_list_add_int(fps_property, label, fps);
	}
	obs_property_set_long_description(fps_property, obs_module_text("NDIPlugin.FilterProps.FpsLimit.Description"));

	obs_property_t *delay_property = obs_properties_add_list(props, FLT_PROP_READBACK_DELAY,
								 obs_module_text("NDIPlugin.FilterProps.ReadbackDelay"),
								 OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(delay_property, obs_module_text("NDIPlugin.FilterProps.ReadbackDelay.Sync"), 0);
	for (int d = 1; d <= MAX_READBACK_DELAY; d++) {
		char label[32];
		snprintf(label, sizeof(label), "%d %s", d,
			 obs_module_text(d == 1 ? "NDIPlugin.FilterProps.ReadbackDelay.Frame"
						: "NDIPlugin.FilterProps.ReadbackDelay.Frames"));
		obs_property_list_add_int(delay_property, label, d);
	}
	obs_property_set_long_description(delay_property,
					  obs_module_text("NDIPlugin.FilterProps.ReadbackDelay.Description"));

	obs_property_t *keep_property = obs_properties_add_bool(props, FLT_PROP_KEEP_ACTIVE,
								obs_module_text("NDIPlugin.FilterProps.KeepActive"));
	obs_property_set_long_description(keep_property,
					  obs_module_text("NDIPlugin.FilterProps.KeepActive.Description"));

	obs_properties_add_button2(
		props, "ndi_apply", obs_module_text("NDIPlugin.FilterProps.ApplySettings"),
		[](obs_properties_t *, obs_property_t *, void *private_data) {
			auto s = (ndi_filter_t *)private_data;
			auto settings = obs_source_get_settings(s->obs_source);
			ndi_filter_update(s, settings);
			obs_data_release(settings);
			return true;
		},
		priv);

	obs_log(LOG_DEBUG, "-ndi_filter_getproperties(...)");
	return props;
}

void ndi_filter_getdefaults(obs_data_t *defaults)
{
	obs_log(LOG_DEBUG, "+ndi_filter_getdefaults(...)");
	obs_data_set_default_string(defaults, FLT_PROP_NAME, obs_module_text("NDIPlugin.FilterProps.NDIName.Default"));
	obs_data_set_default_string(defaults, FLT_PROP_GROUPS, "");
	obs_data_set_default_int(defaults, FLT_PROP_FPS_LIMIT, 0);
	obs_data_set_default_int(defaults, FLT_PROP_READBACK_DELAY, 1);
	obs_data_set_default_bool(defaults, FLT_PROP_KEEP_ACTIVE, true);
	obs_log(LOG_DEBUG, "-ndi_filter_getdefaults(...)");
}

bool is_filter_valid(ndi_filter_t *filter)
{
	obs_source_t *target = obs_filter_get_target(filter->obs_source);
	obs_source_t *parent = obs_filter_get_parent(filter->obs_source);
	if (!target || !parent) {
		return false;
	}

	uint32_t width = obs_source_get_width(filter->obs_source);
	uint32_t height = obs_source_get_height(filter->obs_source);

	// Valid if parent width/height are nonzero, source is enabled, and parent is showing somewhere in OBS's windows
	bool is_valid = (width != 0) && (height != 0) && obs_source_enabled(filter->obs_source) &&
			obs_source_showing(parent);

	return is_valid;
}

void ndi_filter_raw_video(void *data, video_data *frame)
{
	auto f = (ndi_filter_t *)data;

	pthread_mutex_lock(&f->ndi_sender_video_mutex);

	if (!f->ndi_sender) {
		pthread_mutex_unlock(&f->ndi_sender_video_mutex);
		return;
	}

	auto now = std::chrono::steady_clock::now();
	if (now - f->last_video_conn_check >= std::chrono::seconds(1)) {
		f->last_video_conn_check = now;
		int nc = ndiLib->send_get_no_connections(f->ndi_sender, 10);
		if (nc != f->no_video_connections) {
			auto ndi_source = ndiLib->send_get_source_name(f->ndi_sender);
			if (nc <= 0)
				obs_log(LOG_DEBUG, "Dedicated NDI Output video '%s' has no connections",
					ndi_source->p_ndi_name);
			else if (f->no_video_connections == 0)
				obs_log(LOG_DEBUG, "Dedicated NDI Output video '%s' has %d connections",
					ndi_source->p_ndi_name, nc);
			f->no_video_connections = nc;
		}
	}

	pthread_mutex_unlock(&f->ndi_sender_video_mutex);

	NDIlib_video_frame_v2_t video_frame = {0};

	// Only fill out the data if we have a valid frame
	if (frame && frame->data[0]) {
		video_frame.xres = f->known_width;
		video_frame.yres = f->known_height;
		video_frame.FourCC = NDIlib_FourCC_type_BGRA;
		if (f->fps_limit > 0) {
			video_frame.frame_rate_N = f->fps_limit;
			video_frame.frame_rate_D = 1;
		} else {
			video_frame.frame_rate_N = f->ovi.fps_num;
			video_frame.frame_rate_D = f->ovi.fps_den;
		}
		video_frame.picture_aspect_ratio = 0; // square pixels
		video_frame.frame_format_type = NDIlib_frame_format_type_progressive;
#ifdef SYNC_DEBUG
		// Convert OBS timestamp in nanoseconds to NDI timestamp in 100-nanosecond intervals,
		// applying offset to synchronize with audio frames
		video_frame.timestamp = (frame->timestamp + f->obs_to_ndi_time_offset) / 100;
#endif
		video_frame.timecode = NDIlib_send_timecode_synthesize;
		video_frame.p_data = frame->data[0];
		video_frame.line_stride_in_bytes = frame->linesize[0];
	}

	pthread_mutex_lock(&f->ndi_sender_video_mutex);
	SYNC_DEBUG_LOG_VIDEO_TIME("NDI <- ndi_filter", obs_source_get_name(f->obs_source), video_frame.timestamp * 100,
				  (uint8_t *)video_frame.p_data);
	if (f->ndi_sender)
		ndiLib->send_send_video_v2(f->ndi_sender, &video_frame);
	pthread_mutex_unlock(&f->ndi_sender_video_mutex);
}

// ---- async readback worker -------------------------------------------------

static void readback_worker_loop(ndi_filter_t *f)
{
	os_set_thread_name("ndi-filter-readback");
	readback_worker *w = f->worker;
	for (;;) {
		readback_job job;
		{
			std::unique_lock<std::mutex> lk(w->mutex);
			w->cv.wait(lk, [w] { return w->stop || !w->queue.empty(); });
			if (w->queue.empty())
				break; // stop requested and nothing left to send
			job = w->queue.front();
			w->queue.pop_front();
		}

		video_data frame = {};
		frame.data[0] = job.data;
		frame.linesize[0] = job.linesize;
		frame.timestamp = job.timestamp;
		ndi_filter_raw_video(f, &frame); // synchronous NDI send; buffer is free on return

		{
			std::lock_guard<std::mutex> lk(w->mutex);
			f->stage_ring[job.slot].worker_done = true;
			w->in_flight--;
		}
		w->cv.notify_all();
	}
}

static void readback_worker_start(ndi_filter_t *f)
{
	if (f->worker)
		return;
	f->worker = new readback_worker();
	f->worker->thread = std::thread(readback_worker_loop, f);
}

// Wait until the worker has finished every queued send. Graphics thread only.
static void readback_drain(ndi_filter_t *f)
{
	if (!f->worker)
		return;
	std::unique_lock<std::mutex> lk(f->worker->mutex);
	f->worker->cv.wait_for(lk, std::chrono::milliseconds(500), [f] { return f->worker->in_flight == 0; });
}

static void readback_worker_stop(ndi_filter_t *f)
{
	if (!f->worker)
		return;
	{
		std::lock_guard<std::mutex> lk(f->worker->mutex);
		f->worker->stop = true;
	}
	f->worker->cv.notify_all();
	if (f->worker->thread.joinable())
		f->worker->thread.join();
	delete f->worker;
	f->worker = nullptr;
}

// Unmap and free every staging slot. Graphics thread only; drains the worker first.
static void readback_ring_destroy(ndi_filter_t *f)
{
	readback_drain(f);
	for (int i = 0; i < MAX_STAGE_SLOTS; i++) {
		stage_slot &s = f->stage_ring[i];
		if (s.state == STAGE_MAPPED && s.surf)
			gs_stagesurface_unmap(s.surf);
		gs_stagesurface_destroy(s.surf);
		s = {};
	}
	f->ring_size = 0;
	if (f->worker) {
		std::lock_guard<std::mutex> lk(f->worker->mutex);
		f->worker->queue.clear();
	}
}

static void readback_ring_create(ndi_filter_t *f, uint32_t width, uint32_t height)
{
	readback_ring_destroy(f);
	f->ring_size = f->readback_delay + 2;
	for (uint32_t i = 0; i < f->ring_size; i++) {
		f->stage_ring[i].surf = gs_stagesurface_create(width, height, TEXFORMAT);
		f->stage_ring[i].state = STAGE_FREE;
	}
	f->frame_seq = 0;
	f->dropped_logged = false;
}

// Async path: stage this frame, map an older one, hand it to the worker.
static void ndi_filter_readback_async(ndi_filter_t *f, gs_texture_t *tex, uint64_t timestamp)
{
	// 1. Release stage_ring the worker has finished with.
	{
		std::lock_guard<std::mutex> lk(f->worker->mutex);
		for (uint32_t i = 0; i < f->ring_size; i++) {
			stage_slot &s = f->stage_ring[i];
			if (s.state == STAGE_MAPPED && s.worker_done) {
				gs_stagesurface_unmap(s.surf);
				s.state = STAGE_FREE;
				s.worker_done = false;
				s.data = nullptr;
			}
		}
	}

	// 2. Stage the current frame into a free slot (queues the GPU copy, does not wait).
	int free_idx = -1;
	for (uint32_t i = 0; i < f->ring_size; i++) {
		if (f->stage_ring[i].state == STAGE_FREE) {
			free_idx = (int)i;
			break;
		}
	}
	if (free_idx < 0) {
		// Worker is behind (NDI send slower than frame rate); drop this frame rather than stall.
		if (!f->dropped_logged) {
			obs_log(LOG_WARNING, "'%s': readback ring full, dropping frames (NDI send is lagging)",
				obs_source_get_name(f->obs_source));
			f->dropped_logged = true;
		}
		return;
	}
	f->frame_seq++;
	stage_slot &cur = f->stage_ring[free_idx];
	gs_stage_texture(cur.surf, tex);
	cur.state = STAGE_STAGED;
	cur.seq = f->frame_seq;
	cur.timestamp = timestamp;

	// 3. Map the oldest staged slot that is at least `readback_delay` frames old.
	int map_idx = -1;
	for (uint32_t i = 0; i < f->ring_size; i++) {
		stage_slot &s = f->stage_ring[i];
		if (s.state == STAGE_STAGED && s.seq + f->readback_delay <= f->frame_seq &&
		    (map_idx < 0 || s.seq < f->stage_ring[map_idx].seq))
			map_idx = (int)i;
	}
	if (map_idx < 0)
		return;

	stage_slot &old = f->stage_ring[map_idx];
	if (!gs_stagesurface_map(old.surf, &old.data, &old.linesize)) {
		old.state = STAGE_FREE;
		return;
	}
	old.state = STAGE_MAPPED;
	old.worker_done = false;
	{
		std::lock_guard<std::mutex> lk(f->worker->mutex);
		f->worker->queue.push_back({map_idx, old.data, old.linesize, old.timestamp});
		f->worker->in_flight++;
	}
	f->worker->cv.notify_one();
}

void ndi_filter_render_video(void *data, gs_effect_t *)
{
	auto f = (ndi_filter_t *)data;
	obs_source_skip_video_filter(f->obs_source);

	if (f->rendered)
		return;

	obs_source_t *target = obs_filter_get_target(f->obs_source);
	obs_source_t *parent = obs_filter_get_parent(f->obs_source);

	if (!target || !parent) {
		return;
	}

	if (!is_filter_valid(f)) {
		// Send over an empty frame to indicate that the filter is invalid
		ndi_filter_raw_video(data, nullptr);
		return;
	}

	// Frame rate limit: skip the render entirely until the next slot is due.
	const uint64_t now = os_gettime_ns();
	if (f->fps_limit > 0) {
		const uint64_t interval = 1000000000ULL / f->fps_limit;
		if (f->next_frame_ns == 0)
			f->next_frame_ns = now;
		// Quarter-interval tolerance so OBS frame jitter doesn't make us skip a whole extra frame.
		if (now + interval / 4 < f->next_frame_ns) {
			f->rendered = true;
			return;
		}
		f->next_frame_ns += interval;
		if (f->next_frame_ns < now)
			f->next_frame_ns = now + interval;
	}

	uint32_t width = obs_source_get_width(f->obs_source);
	uint32_t height = obs_source_get_height(f->obs_source);

	const bool async = f->readback_delay > 0;
	const bool size_changed = f->known_width != width || f->known_height != height;

	if (async) {
		if (!f->worker)
			readback_worker_start(f);
		if (size_changed || f->ring_size != f->readback_delay + 2)
			readback_ring_create(f, width, height);
	} else if (f->ring_size) {
		readback_ring_destroy(f);
	}

	if (size_changed) {
		gs_stagesurface_destroy(f->stagesurface);
		f->stagesurface = gs_stagesurface_create(width, height, TEXFORMAT);

		video_output_info vi = {0};
		vi.format = VIDEO_FORMAT_BGRA;
		vi.width = width;
		vi.height = height;
		vi.fps_den = f->ovi.fps_den;
		vi.fps_num = f->ovi.fps_num;
		vi.cache_size = 16;
		vi.colorspace = VIDEO_CS_DEFAULT;
		vi.range = VIDEO_RANGE_DEFAULT;
		vi.name = obs_source_get_name(f->obs_source);

		video_output_close(f->video_output);
		video_output_open(&f->video_output, &vi);
		video_output_connect(f->video_output, nullptr, ndi_filter_raw_video, f);

		f->known_width = width;
		f->known_height = height;
	}

	gs_texrender_reset(f->texrender);

	if (gs_texrender_begin(f->texrender, width, height)) {
		vec4 background;
		vec4_zero(&background);

		gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
		gs_ortho(0.0f, (float)width, 0.0f, (float)height, -100.0f, 100.0f);

		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		if (target == parent) {
			obs_source_skip_video_filter(f->obs_source);
		} else {
			obs_source_video_render(target);
		}

		gs_blend_state_pop();
		gs_texrender_end(f->texrender);

		if (async) {
			ndi_filter_readback_async(f, gs_texrender_get_texture(f->texrender), now);
			f->rendered = true;
			return;
		}

		gs_stage_texture(f->stagesurface, gs_texrender_get_texture(f->texrender));
		if (gs_stagesurface_map(f->stagesurface, &f->video_data, &f->video_linesize)) {
			video_frame output_frame;
			if (video_output_lock_frame(f->video_output, &output_frame, 1, now)) {
				uint32_t linesize = output_frame.linesize[0];
				for (uint32_t i = 0; i < f->known_height; ++i) {
					uint32_t dst_offset = linesize * i;
					uint32_t src_offset = f->video_linesize * i;
					memcpy(output_frame.data[0] + dst_offset, f->video_data + src_offset, linesize);
				}

				video_output_unlock_frame(f->video_output);
			}

			gs_stagesurface_unmap(f->stagesurface);
		}
	}

	f->rendered = true;
}

void ndi_sender_destroy(ndi_filter_t *filter)
{
	if (!filter || !filter->ndi_sender) {
		return;
	}

	if (!filter->is_audioonly) {
		pthread_mutex_lock(&filter->ndi_sender_video_mutex);
	}

	pthread_mutex_lock(&filter->ndi_sender_audio_mutex);
	ndiLib->send_destroy(filter->ndi_sender);
	filter->ndi_sender = nullptr;
	pthread_mutex_unlock(&filter->ndi_sender_audio_mutex);

	if (!filter->is_audioonly) {
		pthread_mutex_unlock(&filter->ndi_sender_video_mutex);
	}
}

extern void replace_invalid_filename_chars(QString *s); // defined in forms/output-settings.cpp

void ndi_sender_create(ndi_filter_t *filter, obs_data_t *settings)
{
	bool allocating_settings = false;

	if (!filter || !filter->obs_source) {
		return;
	}

	auto obs_source = filter->obs_source;
	if (!settings) {
		settings = obs_source_get_settings(obs_source);
		allocating_settings = true;
	}

	auto parent_source = obs_filter_get_parent(filter->obs_source);

	if (!parent_source) {
		// Don't create sender if the parent source is not available
		// It will be created in filter_tick
		if (allocating_settings)
			obs_data_release(settings);
		return;
	}

	QString ndi_name = obs_data_get_string(settings, FLT_PROP_NAME);

	// Replace any invalid filename characters in the original NDI name before saving it back to settings
	replace_invalid_filename_chars(&ndi_name);
	obs_data_set_string(settings, FLT_PROP_NAME, ndi_name.toUtf8().constData());

	// Check the original template for tokens before any replacements are made,
	// so injected source/filter names cannot trigger unintended token expansion.
	const bool has_source_token = ndi_name.contains("${source}");
	const bool has_filter_token = ndi_name.contains("${filter}");

	// If the name contains ${source} then replace it with the source name.
	if (has_source_token) {
		auto parent_name = obs_source_get_name(parent_source);
		ndi_name.replace("${source}", QString(parent_name));
		signal_handler_t *sh = obs_source_get_signal_handler(parent_source);
		if (sh) {
			signal_handler_disconnect(sh, "rename", on_renamed, filter);
			signal_handler_connect(sh, "rename", on_renamed, filter);
		}
	}

	// If the name contains ${filter} then replace it with the filter name.
	if (has_filter_token) {
		auto filter_name = obs_source_get_name(filter->obs_source);
		ndi_name.replace("${filter}", QString(filter_name));
		signal_handler_t *sh = obs_source_get_signal_handler(filter->obs_source);
		if (sh) {
			signal_handler_disconnect(sh, "rename", on_renamed, filter);
			signal_handler_connect(sh, "rename", on_renamed, filter);
		}
	}

	// Replace any invalid filename characters in the final NDI name
	replace_invalid_filename_chars(&ndi_name);

	QByteArray ndi_name_utf8 = ndi_name.toUtf8();

	NDIlib_send_create_t send_desc{};
	send_desc.p_ndi_name = ndi_name_utf8.constData();

	auto groups = obs_data_get_string(settings, FLT_PROP_GROUPS);

	if (groups && groups[0])
		send_desc.p_groups = groups;
	else
		send_desc.p_groups = nullptr;
	send_desc.clock_video = false;
	send_desc.clock_audio = false;

	if (!filter->is_audioonly) {
		pthread_mutex_lock(&filter->ndi_sender_video_mutex);
	}

	pthread_mutex_lock(&filter->ndi_sender_audio_mutex);
	ndiLib->send_destroy(filter->ndi_sender);
	filter->ndi_sender = ndiLib->send_create(&send_desc);

	if (filter->ndi_sender) {
		obs_log(LOG_INFO, "Dedicated NDI Output sender created: '%s'", send_desc.p_ndi_name);
		obs_log(LOG_DEBUG, "'%s' ndi_sender_create: ndi output started", send_desc.p_ndi_name);
	} else {
		obs_log(LOG_WARNING, "WARN-416 - Dedicated NDI Output sender initialisation failed. '%s'",
			send_desc.p_ndi_name);
		obs_log(LOG_DEBUG, "'%s' ndi_sender_create: ndi sender init failed", send_desc.p_ndi_name);
	}

	filter->no_audio_connections = -1;

#ifdef SYNC_DEBUG
	// Video timestamps are in OBS time, so calculate offset to convert to NDI timestamps
	filter->obs_to_ndi_time_offset = now_ns() - os_gettime_ns();
#endif

	pthread_mutex_unlock(&filter->ndi_sender_audio_mutex);

	if (!filter->is_audioonly) {
		pthread_mutex_unlock(&filter->ndi_sender_video_mutex);
	}

	if (allocating_settings) {
		obs_data_release(settings);
	}
}

void ndi_filter_update(void *data, obs_data_t *settings)
{
	auto f = (ndi_filter_t *)data;
	auto obs_source = f->obs_source;
	auto name = obs_source_get_name(obs_source);
	obs_log(LOG_DEBUG, "+ndi_filter_update(name='%s')", name);

	ndi_sender_create(f, settings);

	auto groups = obs_data_get_string(settings, FLT_PROP_GROUPS);

	if (!f->is_audioonly) {
		long long fps = obs_data_get_int(settings, FLT_PROP_FPS_LIMIT);
		long long delay = obs_data_get_int(settings, FLT_PROP_READBACK_DELAY);
		if (fps < 0)
			fps = 0;
		if (delay < 0)
			delay = 0;
		if (delay > MAX_READBACK_DELAY)
			delay = MAX_READBACK_DELAY;
		f->fps_limit = (uint32_t)fps;
		f->next_frame_ns = 0;
		// The ring is (re)built on the graphics thread in render_video when this changes.
		f->readback_delay = (uint32_t)delay;
		f->keep_active = obs_data_get_bool(settings, FLT_PROP_KEEP_ACTIVE);
		ndi_filter_apply_keep_active(f);
	}

	obs_log(LOG_INFO, "NDI Filter Updated: '%s' (fps limit %u, readback delay %u)", name, f->fps_limit,
		f->readback_delay);
	obs_log(LOG_DEBUG, "-ndi_filter_update(name='%s', groups='%s')", name, groups);
}

void *ndi_filter_create(obs_data_t *settings, obs_source_t *obs_source)
{
	auto name = obs_data_get_string(settings, FLT_PROP_NAME);
	auto groups = obs_data_get_string(settings, FLT_PROP_GROUPS);
	obs_log(LOG_DEBUG, "+ndi_filter_create(name='%s', groups='%s')", name, groups);

	auto f = (ndi_filter_t *)bzalloc(sizeof(ndi_filter_t));
	f->obs_source = obs_source;
	f->texrender = gs_texrender_create(TEXFORMAT, GS_ZS_NONE);
	pthread_mutex_init(&f->ndi_sender_video_mutex, NULL);
	pthread_mutex_init(&f->ndi_sender_audio_mutex, NULL);
	obs_get_video_info(&f->ovi);
	obs_get_audio_info(&f->oai);

	ndi_filter_update(f, settings);

	obs_log(LOG_INFO, "NDI Filter Created: '%s'", name);
	obs_log(LOG_DEBUG, "-ndi_filter_create(...)");

	return f;
}

void *ndi_filter_create_audioonly(obs_data_t *settings, obs_source_t *obs_source)
{
	auto name = obs_data_get_string(settings, FLT_PROP_NAME);
	auto groups = obs_data_get_string(settings, FLT_PROP_GROUPS);
	obs_log(LOG_DEBUG, "+ndi_filter_create_audioonly(name='%s', groups='%s')", name, groups);

	auto f = (ndi_filter_t *)bzalloc(sizeof(ndi_filter_t));
	f->is_audioonly = true;
	f->obs_source = obs_source;
	pthread_mutex_init(&f->ndi_sender_audio_mutex, NULL);
	obs_get_audio_info(&f->oai);

	ndi_filter_update(f, settings);

	obs_log(LOG_INFO, "NDI Audio-Only Filter Created: '%s'", name);
	obs_log(LOG_DEBUG, "-ndi_filter_create_audioonly(...)");

	return f;
}

void ndi_filter_destroy(void *data)
{
	auto f = (ndi_filter_t *)data;
	auto name = obs_source_get_name(f->obs_source);
	obs_log(LOG_DEBUG, "+ndi_filter_destroy('%s'...)", name);

	ndi_filter_release_active(f);
	ndi_filter_disconnect_rename_handlers(f);

	video_output_close(f->video_output);

	// Let the worker finish any in-flight send, then stop it, before the sender and surfaces go.
	readback_drain(f);
	readback_worker_stop(f);

	ndi_sender_destroy(f);

	obs_enter_graphics();
	readback_ring_destroy(f);
	gs_stagesurface_unmap(f->stagesurface);
	gs_stagesurface_destroy(f->stagesurface);
	gs_texrender_destroy(f->texrender);
	obs_leave_graphics();

	if (f->audio_conv_buffer) {
		obs_log(LOG_DEBUG, "ndi_filter_destroy: freeing %zu bytes", f->audio_conv_buffer_size);
		bfree(f->audio_conv_buffer);
		f->audio_conv_buffer = nullptr;
	}

	bfree(f);

	obs_log(LOG_INFO, "NDI Filter Destroyed: '%s'", name);
	obs_log(LOG_DEBUG, "-ndi_filter_destroy('%s'...)", name);
}

void ndi_filter_destroy_audioonly(void *data)
{
	auto f = (ndi_filter_t *)data;
	auto name = obs_source_get_name(f->obs_source);
	obs_log(LOG_DEBUG, "+ndi_filter_destroy_audioonly('%s'...)", name);

	ndi_filter_disconnect_rename_handlers(f);
	ndi_sender_destroy(f);

	if (f->audio_conv_buffer) {
		bfree(f->audio_conv_buffer);
		f->audio_conv_buffer = nullptr;
	}

	bfree(f);

	obs_log(LOG_INFO, "NDI Audio-Only Filter Destroyed: '%s'", name);
	obs_log(LOG_DEBUG, "-ndi_filter_destroy_audioonly('%s'...)", name);
}

void ndi_filter_tick(void *data, float)
{
	auto f = (ndi_filter_t *)data;

	obs_get_video_info(&f->ovi);

	f->rendered = false;
}

void ndi_filter_add(void *data, obs_source_t * /* parent */)
{
	auto f = (ndi_filter_t *)data;
	if (!f->ndi_sender)
		ndi_sender_create(f, nullptr);
	ndi_filter_apply_keep_active(f);
}

void ndi_filter_remove(void *data, obs_source_t * /* parent */)
{
	auto f = (ndi_filter_t *)data;
	if (!f)
		return;

	ndi_filter_release_active(f);
	ndi_filter_disconnect_rename_handlers(f);
	ndi_sender_destroy(f);
}

obs_audio_data *ndi_filter_asyncaudio(void *data, obs_audio_data *audio_data)
{
	// NOTE: The logic in this function should be similar to
	// ndi-output.cpp/ndi_output_raw_audio(...)
	auto f = (ndi_filter_t *)data;

	pthread_mutex_lock(&f->ndi_sender_audio_mutex);

	if (!f->ndi_sender) {
		pthread_mutex_unlock(&f->ndi_sender_audio_mutex);
		return audio_data;
	}

	auto now = std::chrono::steady_clock::now();
	if (now - f->last_audio_conn_check >= std::chrono::seconds(1)) {
		f->last_audio_conn_check = now;
		int nc = ndiLib->send_get_no_connections(f->ndi_sender, 10);
		if (nc != f->no_audio_connections) {
			auto ndi_source = ndiLib->send_get_source_name(f->ndi_sender);
			if (nc <= 0)
				obs_log(LOG_DEBUG, "Dedicated NDI Output audio '%s' has no connections.",
					ndi_source->p_ndi_name);
			else if (f->no_audio_connections == 0)
				obs_log(LOG_DEBUG, "Dedicated NDI Output audio '%s' has %d connections.",
					ndi_source->p_ndi_name, nc);
			f->no_audio_connections = nc;
		}
	}

	pthread_mutex_unlock(&f->ndi_sender_audio_mutex);

	obs_get_audio_info(&f->oai);

	NDIlib_audio_frame_v3_t audio_frame = {0};
	audio_frame.sample_rate = f->oai.samples_per_sec;
	audio_frame.no_channels = f->oai.speakers;
#ifdef SYNC_DEBUG
	audio_frame.timestamp = audio_data->timestamp / 100;
#endif
	audio_frame.timecode = NDIlib_send_timecode_synthesize;
	audio_frame.no_samples = audio_data->frames;
	audio_frame.channel_stride_in_bytes =
		audio_frame.no_samples *
		4; // TODO: Check if this correct or should 4 be replaced by number of channels.
	// audio_frame.FourCC = NDIlib_FourCC_audio_type_FLTP;
	// audio_frame.p_data = p_frame;
	audio_frame.p_metadata = NULL; // No metadata support yet!

	const size_t data_size =
		static_cast<size_t>(audio_frame.no_channels) * static_cast<size_t>(audio_frame.channel_stride_in_bytes);

	if (data_size > f->audio_conv_buffer_size) {
		obs_log(LOG_DEBUG, "ndi_filter_asyncaudio: growing audio_conv_buffer from %zu to %zu bytes",
			f->audio_conv_buffer_size, data_size);
		if (f->audio_conv_buffer) {
			obs_log(LOG_DEBUG, "ndi_filter_asyncaudio: freeing %zu bytes", f->audio_conv_buffer_size);
			bfree(f->audio_conv_buffer);
		}
		obs_log(LOG_DEBUG, "ndi_filter_asyncaudio: allocating %zu bytes", data_size);
		f->audio_conv_buffer = (uint8_t *)bmalloc(data_size);
		f->audio_conv_buffer_size = data_size;
	}

	for (int i = 0; i < audio_frame.no_channels; ++i) {
		memcpy(f->audio_conv_buffer + (i * audio_frame.channel_stride_in_bytes), audio_data->data[i],
		       audio_frame.channel_stride_in_bytes);
	}

	audio_frame.p_data = f->audio_conv_buffer;

	pthread_mutex_lock(&f->ndi_sender_audio_mutex);
	SYNC_DEBUG_LOG_AUDIO_TIME("NDI <- ndi_filter", obs_source_get_name(f->obs_source), audio_frame.timestamp * 100,
				  (float *)audio_frame.p_data, audio_frame.no_samples, audio_frame.sample_rate);
	if (f->ndi_sender)
		ndiLib->send_send_audio_v3(f->ndi_sender, &audio_frame);
	pthread_mutex_unlock(&f->ndi_sender_audio_mutex);

	return audio_data;
}

obs_source_info create_ndi_filter_info()
{
	obs_source_info ndi_filter_info = {};
	ndi_filter_info.id = "ndi_filter";
	ndi_filter_info.type = OBS_SOURCE_TYPE_FILTER;
	ndi_filter_info.output_flags = OBS_SOURCE_VIDEO;

	ndi_filter_info.get_name = ndi_filter_getname;
	ndi_filter_info.get_properties = ndi_filter_getproperties;
	ndi_filter_info.get_defaults = ndi_filter_getdefaults;

	ndi_filter_info.create = ndi_filter_create;
	ndi_filter_info.filter_add = ndi_filter_add;
	ndi_filter_info.filter_remove = ndi_filter_remove;
	ndi_filter_info.destroy = ndi_filter_destroy;
	ndi_filter_info.update = ndi_filter_update;

	ndi_filter_info.video_tick = ndi_filter_tick;
	ndi_filter_info.video_render = ndi_filter_render_video;

	// Audio is available only with async sources
	ndi_filter_info.filter_audio = ndi_filter_asyncaudio;

	return ndi_filter_info;
}

obs_source_info create_ndi_audiofilter_info()
{
	obs_source_info ndi_filter_info = {};
	ndi_filter_info.id = "ndi_audiofilter";
	ndi_filter_info.type = OBS_SOURCE_TYPE_FILTER;
	ndi_filter_info.output_flags = OBS_SOURCE_AUDIO;

	ndi_filter_info.get_name = ndi_audiofilter_getname;
	ndi_filter_info.get_properties = ndi_filter_getproperties;
	ndi_filter_info.get_defaults = ndi_filter_getdefaults;

	ndi_filter_info.create = ndi_filter_create_audioonly;
	ndi_filter_info.filter_add = ndi_filter_add;
	ndi_filter_info.filter_remove = ndi_filter_remove;
	ndi_filter_info.update = ndi_filter_update;
	ndi_filter_info.destroy = ndi_filter_destroy_audioonly;

	ndi_filter_info.filter_audio = ndi_filter_asyncaudio;

	return ndi_filter_info;
}
