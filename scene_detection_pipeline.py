from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import logging
import math
import re
import subprocess
import time
import urllib.request
import wave
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from srt_utils import SrtSegment, parse_srt
from time_utils import format_timestamp


class FaceAnalysisError(RuntimeError):
    pass


@dataclass
class Segment:
    start: float
    end: float


@dataclass
class SceneDetectionConfig:
    sample_rate: int = 16000
    rms_frame_ms: int = 50
    vad_pause_min: float = 0.3
    vad_merge_gap: float = 0.6
    vad_min_speech: float = 0.15
    vad_gap_fill_ms: float = 150.0
    vad_threshold_p50: float = 0.8
    vad_threshold_p80: float = 0.3
    min_clip_seconds: float = 30.0
    max_clip_seconds: float = 300.0
    max_blocks_per_candidate: int = 3
    peak_percentile: float = 85.0
    dedup_iou: float = 0.6
    top_k_min: int = 3
    top_k_max: int = 7
    whisperx_enabled: bool = True
    whisperx_pad_seconds: float = 3.0
    whisperx_env_name: Optional[str] = None
    whisperx_require_env: bool = False
    snap_window_seconds: float = 0.8
    start_offset_seconds: float = 0.12
    end_offset_seconds: float = 0.15
    start_offset_seconds_no_align: float = 0.05
    end_offset_seconds_no_align: float = 0.35
    face_enabled: bool = True
    face_required: bool = True
    face_target_fps: float = 2.0
    face_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    face_model_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
        / "assets"
        / "mediapipe"
        / "face_landmarker.task"
    )
    transcript_start_step: int = 1
    transcript_extend_step: int = 3
    transcript_max_candidates: int = 300
    transcript_require_sentence_end: bool = False
    transcript_relax_sentence_end_if_empty: bool = True
    speech_ratio_min: float = 0.2
    whisper_bad_ratio_max: float = 0.8
    words_per_second_min: float = 0.5
    incomplete_thought_penalty: float = 0.08
    weight_text: float = 0.40
    weight_cut: float = 0.10
    weight_audio: float = 0.20
    weight_face: float = 0.30


class SceneDetectionPipeline:
    def __init__(
        self,
        context_dir: Path,
        config: Optional[SceneDetectionConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.context_dir = Path(context_dir)
        self.config = config or SceneDetectionConfig()
        self.logger = logger or logging.getLogger("ai_videoclipper")

        self.artifacts_dir = self.context_dir / "scene_detection"
        self._artifact_base: Optional[str] = None
        self._whisperx_model = None
        self._whisperx_align_model = None
        self._whisperx_metadata = None
        self._whisperx_device = None
        self._whisperx_language = None
        if isinstance(self.config.face_model_path, str):
            self.config.face_model_path = Path(self.config.face_model_path)
        self._analysis_meta_path = self.artifacts_dir / "analysis_meta.json"
        self._analysis_timestamp_path = self.artifacts_dir / "analysis_timestamp.json"
        self._analysis_status_path = self.artifacts_dir / "analysis_status.json"

    def prepare_candidates(
        self,
        video_path: Path,
        srt_path: Path,
    ) -> Dict[str, object]:
        self._set_artifacts_dir(video_path)
        self.logger.info("[SCENE] Preparing candidates...")
        audio = self._build_audio_pack(video_path)
        transcript = self._build_transcript_pack(srt_path)
        face = self._build_face_pack(video_path)
        self.logger.info(
            "[SCENE] Packs ready: audio=%s, transcript=%s, face=%s",
            bool(audio),
            bool(transcript),
            bool(face),
        )

        candidates = self._generate_candidates(
            audio["speech_segments"],
            audio["pauses"],
            face["series"] if face else None,
            transcript["segments"],
        )
        enriched = self._enrich_candidates(candidates, audio, transcript, face)
        self.logger.info(
            "[SCENE] Candidates: raw=%d, enriched=%d",
            len(candidates),
            len(enriched),
        )

        bundle = {
            "audio": audio,
            "transcript": transcript,
            "face": face,
            "candidates": candidates,
            "enriched_candidates": enriched,
        }

        self._write_json(self._artifact_path("analysis_audio_rms"), audio["rms_series"])
        self._write_json(self._artifact_path("analysis_audio_vad"), audio["speech_segments"])
        self._write_json(self._artifact_path("analysis_audio_pauses"), audio["pauses"])
        self._write_json(self._artifact_path("analysis_audio_pack"), self._audio_pack_meta(audio))
        self._write_analysis_status()  # Save status after audio artifacts
        if face:
            self._write_json(self._artifact_path("analysis_face"), face["series"])
            self._write_analysis_status()  # Save status after face artifacts
        self._write_analysis_meta(video_path, srt_path)
        self._write_json(self._artifact_path("analysis_clip_candidates"), candidates)
        self._write_json(self._artifact_path("analysis_clip_candidates_enriched"), enriched)
        self._write_analysis_status()  # Final status update
        self.logger.info("[SCENE] Candidate artifacts written to %s", self.artifacts_dir)
        return bundle

    def prepare_candidates_from_list(
        self,
        video_path: Path,
        srt_path: Path,
        candidates: List[Dict[str, object]],
    ) -> Dict[str, object]:
        self._set_artifacts_dir(video_path)
        self.logger.info("[SCENE] Preparing candidates from LLM discovery (%d)", len(candidates))
        audio = self._build_audio_pack(video_path)
        transcript = self._build_transcript_pack(srt_path)
        face = self._build_face_pack(video_path)

        normalized = self._normalize_candidate_list(candidates)
        enriched = self._enrich_candidates(normalized, audio, transcript, face)

        bundle = {
            "audio": audio,
            "transcript": transcript,
            "face": face,
            "candidates": normalized,
            "enriched_candidates": enriched,
        }

        # Save audio artifacts
        self._write_json(self._artifact_path("analysis_audio_rms"), audio["rms_series"])
        self._write_json(self._artifact_path("analysis_audio_vad"), audio["speech_segments"])
        self._write_json(self._artifact_path("analysis_audio_pauses"), audio["pauses"])
        self._write_json(self._artifact_path("analysis_audio_pack"), self._audio_pack_meta(audio))
        self._write_analysis_status()  # Save status after audio artifacts

        # Save face artifacts if available
        if face:
            self._write_json(self._artifact_path("analysis_face"), face["series"])
            self._write_analysis_status()  # Save status after face artifacts

        # Save candidate artifacts
        self._write_json(self._artifact_path("analysis_clip_candidates_discovery"), candidates)
        self._write_json(self._artifact_path("analysis_clip_candidates"), normalized)
        self._write_json(self._artifact_path("analysis_clip_candidates_enriched"), enriched)
        self._write_analysis_meta(video_path, srt_path)
        self._write_analysis_status()  # Final status update
        self.logger.info("[SCENE] Candidate artifacts written to %s", self.artifacts_dir)
        return bundle

    def score_and_rank(
        self,
        bundle: Dict[str, object],
        llm_scores: List[Dict[str, object]],
    ) -> Dict[str, object]:
        self.logger.info("[SCENE] Scoring %d candidates with LLM results", len(llm_scores))
        enriched = bundle["enriched_candidates"]
        scored = self._apply_llm_scores(enriched, llm_scores)
        ranked = self._rank_candidates(scored)
        final_candidates = self._apply_diversity(ranked)
        if len(final_candidates) < self.config.top_k_min:
            final_candidates = ranked[: self.config.top_k_min]
        final_candidates = final_candidates[: self.config.top_k_max]
        self.logger.info(
            "[SCENE] Ranking complete: scored=%d, ranked=%d, final=%d",
            len(scored),
            len(ranked),
            len(final_candidates),
        )

        self._write_json(self._artifact_path("analysis_llm_scores"), llm_scores)
        self._write_json(self._artifact_path("analysis_top_candidates"), final_candidates)
        self._write_analysis_status()

        return {
            "ranked_candidates": ranked,
            "top_candidates": final_candidates,
        }

    def refine_cuts(
        self,
        bundle: Dict[str, object],
        top_candidates: List[Dict[str, object]],
        video_path: Path,
    ) -> List[Dict[str, object]]:
        self.logger.info("[SCENE] Refining cuts for %d candidates", len(top_candidates))
        if not top_candidates:
            self.logger.info("[SCENE] WhisperX alignment skipped (no top candidates)")
        audio = bundle["audio"]
        pauses = [Segment(**pause) for pause in audio["pauses"]]
        transcript_segments = bundle["transcript"]["segments"]
        face_series = bundle.get("face", {}).get("series") if bundle.get("face") else None
        refined: List[Dict[str, object]] = []
        for candidate in top_candidates:
            candidate_id = candidate["candidate_id"]
            start = float(candidate["start"])
            end = float(candidate["end"])
            # Skip WhisperX alignment for now - will be done at export time
            # alignment = self._whisperx_align_candidate(
            #     video_path,
            #     candidate_id,
            #     start,
            #     end,
            #     transcript_segments,
            # )
            alignment = None  # Use fallback offsets instead of word-level alignment
            final_start, final_end, meta = self._refine_boundaries(
                start,
                end,
                pauses,
                alignment,
            )
            key_moment = self._compute_key_moment(
                final_start,
                final_end,
                face_series,
                audio["rms_series"],
            )
            refined.append(
                {
                    **candidate,
                    "start": final_start,
                    "end": final_end,
                    "cut_metadata": meta,
                    "key_moment": key_moment,
                }
            )

        self._write_json(self._artifact_path("analysis_cutlist"), refined)
        self._write_analysis_status()
        self.logger.info("[SCENE] Cutlist written (%d clips)", len(refined))
        return refined

    def build_clip_config(self, candidates: List[Dict[str, object]]) -> Dict[str, object]:
        self.logger.info("[SCENE] Building clips config for %d clips", len(candidates))
        clips = []
        for idx, candidate in enumerate(candidates, start=1):
            title = candidate.get("title") or f"Clip {idx}"
            summary = candidate.get("summary", "").strip()
            reason = candidate.get("reason", "").strip()
            key_moment = candidate.get("key_moment") or {}
            name_parts = [title.strip()]
            if summary:
                name_parts.append(f"\n\nZusammenfassung: {summary}")
            if reason:
                name_parts.append(f"\n\nBegründung: {reason}")
            if key_moment and key_moment.get("start") is not None and key_moment.get("end") is not None:
                start_ts = format_timestamp(float(key_moment["start"]))
                end_ts = format_timestamp(float(key_moment["end"]))
                name_parts.append(f"\n\nKey-Moment: {start_ts}–{end_ts}")
            name = "".join(name_parts)
            clips.append(
                {
                    "name": name,
                    "start_time": round(float(candidate["start"]), 3),
                    "end_time": round(float(candidate["end"]), 3),
                }
            )
        return {
            "mode": "manual",
            "selection_type": "time",
            "clips": clips,
        }

    def _build_audio_pack(self, video_path: Path) -> Dict[str, object]:
        cached = self._load_cached_audio_pack(video_path)
        if cached:
            self.logger.info("[SCENE] Audio pack: loaded from cache")
            return cached
        self.logger.info("[SCENE] Audio pack: extracting audio")
        audio_path = self._artifact_path("analysis_audio", ext=".wav")
        self._extract_audio(video_path, audio_path)
        audio, sample_rate = self._load_audio(audio_path)
        self.logger.info("[SCENE] Audio pack: loaded (%d samples @ %d Hz)", len(audio), sample_rate)
        frame_sec = self.config.rms_frame_ms / 1000.0
        rms_series = self._compute_rms(audio, sample_rate, frame_sec)

        rms_values = np.array([item["rms"] for item in rms_series], dtype=np.float32)
        rms_stats = {
            "p50": float(np.percentile(rms_values, 50)) if rms_values.size else 0.0,
            "p60": float(np.percentile(rms_values, 60)) if rms_values.size else 0.0,
            "p80": float(np.percentile(rms_values, 80)) if rms_values.size else 0.0,
            "p90": float(np.percentile(rms_values, 90)) if rms_values.size else 0.0,
        }

        speech_segments = self._compute_vad_segments(
            rms_series,
            frame_sec,
            rms_stats,
        )
        duration = self._audio_duration(audio, sample_rate)
        speech_duration = sum(seg.end - seg.start for seg in speech_segments)
        if duration > 0:
            self.logger.info(
                "[SCENE] Audio pack: speech_ratio_total=%.3f",
                speech_duration / duration,
            )
        pauses = self._derive_pauses(speech_segments, duration)
        self.logger.info(
            "[SCENE] Audio pack: rms=%d frames, speech=%d, pauses=%d",
            len(rms_series),
            len(speech_segments),
            len(pauses),
        )

        return {
            "audio_path": str(audio_path),
            "sample_rate": sample_rate,
            "frame_sec": frame_sec,
            "rms_series": rms_series,
            "rms_stats": rms_stats,
            "speech_segments": [seg.__dict__ for seg in speech_segments],
            "pauses": [seg.__dict__ for seg in pauses],
            "duration": duration,
        }

    def _build_transcript_pack(self, srt_path: Path) -> Dict[str, object]:
        self.logger.info("[SCENE] Transcript pack: loading SRT %s", srt_path)
        segments = parse_srt(str(srt_path))
        self.logger.info("[SCENE] Transcript pack: %d segments", len(segments))
        transcript_path = srt_path.with_suffix(".json")
        transcript_json = None
        if transcript_path.exists():
            try:
                transcript_json = json.loads(transcript_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.logger.warning("[SCENE] Transcript JSON exists but failed to parse")
        return {
            "segments": segments,
            "transcript_json": transcript_json,
        }

    def _build_face_pack(self, video_path: Path) -> Optional[Dict[str, object]]:
        if not self.config.face_enabled:
            self.logger.info("[SCENE] Face pack disabled by configuration")
            return None
        cached = self._load_cached_face_pack(video_path)
        if cached:
            self.logger.info("[SCENE] Face pack: loaded from cache")
            return cached
        self.logger.info("[SCENE] Face pack: starting analysis")
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore
        except Exception as exc:
            message = "Face analysis unavailable (mediapipe/cv2 not installed)"
            if self.config.face_required:
                raise FaceAnalysisError(message) from exc
            self.logger.info(f"[SCENE] {message}")
            return None

        model_path = self._ensure_face_model()

        try:
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            message = f"Face landmarker init failed: {exc}"
            if self.config.face_required:
                raise FaceAnalysisError(message) from exc
            self.logger.info(f"[SCENE] {message}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            message = "Face analysis failed to open video"
            if self.config.face_required:
                raise FaceAnalysisError(message)
            self.logger.info(f"[SCENE] {message}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_fps = max(self.config.face_target_fps, 0.1)
        step = max(int(round(fps / target_fps)), 1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_samples = math.ceil(total_frames / step) if total_frames > 0 else None
        frame_idx = 0
        processed = 0
        last_bucket = -5
        last_log_time = time.monotonic()
        series: List[Dict[str, float]] = []

        if total_samples:
            self.logger.info("[SCENE] Face pack: targeting %d sampled frames", total_samples)

        prev_landmarks: Optional[np.ndarray] = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % step != 0:
                    frame_idx += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect(image)
                timestamp = frame_idx / fps

                if not results.face_landmarks:
                    series.append(
                        {
                            "t": timestamp,
                            "face_ok": 0.0,
                            "expressivity": 0.0,
                            "mouth_open": 0.0,
                            "smile_index": 0.0,
                            "brow_raise": 0.0,
                            "eye_open": 0.0,
                            "pose_yaw": 0.0,
                            "pose_pitch": 0.0,
                            "pose_roll": 0.0,
                        }
                    )
                    processed += 1
                    last_bucket, last_log_time = self._log_face_progress(
                        processed,
                        total_samples,
                        last_bucket,
                        last_log_time,
                    )
                    frame_idx += 1
                    continue

                landmarks = results.face_landmarks[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                if prev_landmarks is None:
                    delta = 0.0
                else:
                    delta = float(np.mean(np.linalg.norm(coords - prev_landmarks, axis=1)))
                prev_landmarks = coords

                blendshape_scores = self._blendshape_scores(results)
                blendshape_features = self._blendshape_features(blendshape_scores)
                yaw, pitch, roll = self._pose_from_matrix(results)

                series.append(
                    {
                        "t": timestamp,
                        "face_ok": 1.0,
                        "expressivity": delta,
                        "mouth_open": blendshape_features["mouth_open"],
                        "smile_index": blendshape_features["smile_index"],
                        "brow_raise": blendshape_features["brow_raise"],
                        "eye_open": blendshape_features["eye_open"],
                        "pose_yaw": yaw,
                        "pose_pitch": pitch,
                        "pose_roll": roll,
                    }
                )
                processed += 1
                last_bucket, last_log_time = self._log_face_progress(
                    processed,
                    total_samples,
                    last_bucket,
                    last_log_time,
                )
                frame_idx += 1
        finally:
            cap.release()
            try:
                landmarker.close()
            except Exception:
                pass

        if not series:
            message = "Face analysis produced no frames"
            if self.config.face_required:
                raise FaceAnalysisError(message)
            self.logger.info(f"[SCENE] {message}")
            return None
        self.logger.info("[SCENE] Face pack: %d frames analyzed", len(series))

        values = np.array([item["expressivity"] for item in series], dtype=np.float32)
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std()) if values.size else 1.0
        std = std if std > 1e-6 else 1.0
        for item in series:
            item["expressivity_z"] = (item["expressivity"] - mean) / std

        return {"series": series}

    def _ensure_face_model(self) -> Path:
        model_path = self.config.face_model_path
        if model_path.exists():
            self.logger.info("[SCENE] Face model found at %s", model_path)
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.logger.info(f"[SCENE] Downloading face model to {model_path}")
            urllib.request.urlretrieve(self.config.face_model_url, model_path)
        except Exception as exc:
            message = f"Face model download failed: {exc}"
            if self.config.face_required:
                raise FaceAnalysisError(message) from exc
            self.logger.info(f"[SCENE] {message}")
        return model_path

    def _set_artifacts_dir(self, video_path: Path) -> None:
        self.artifacts_dir = video_path.parent
        self._artifact_base = video_path.stem
        self._analysis_meta_path = self._artifact_path("analysis_meta")
        self._analysis_timestamp_path = self._artifact_path("analysis_timestamp")
        self._analysis_status_path = self._artifact_path("analysis_status")

    def _artifact_path(self, suffix: str, ext: str = ".json") -> Path:
        base = self._artifact_base or "analysis"
        return self.artifacts_dir / f"{base}_{suffix}{ext}"

    def _audio_pack_meta(self, audio: Dict[str, object]) -> Dict[str, object]:
        return {
            "audio_path": audio.get("audio_path"),
            "sample_rate": audio.get("sample_rate"),
            "frame_sec": audio.get("frame_sec"),
            "duration": audio.get("duration"),
            "rms_stats": audio.get("rms_stats"),
        }

    def _load_cached_audio_pack(self, video_path: Path) -> Optional[Dict[str, object]]:
        if not self._is_cache_valid(video_path):
            return None
        required = [
            "analysis_audio_rms",
            "analysis_audio_vad",
            "analysis_audio_pauses",
            "analysis_audio_pack",
        ]
        if not all(self._is_artifact_complete(suffix) for suffix in required):
            return None
        rms_path = self._artifact_path("analysis_audio_rms")
        vad_path = self._artifact_path("analysis_audio_vad")
        pauses_path = self._artifact_path("analysis_audio_pauses")
        pack_path = self._artifact_path("analysis_audio_pack")
        try:
            rms_series = json.loads(rms_path.read_text(encoding="utf-8"))
            speech_segments = json.loads(vad_path.read_text(encoding="utf-8"))
            pauses = json.loads(pauses_path.read_text(encoding="utf-8"))
            pack_meta = json.loads(pack_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return {
            "audio_path": pack_meta.get("audio_path"),
            "sample_rate": pack_meta.get("sample_rate"),
            "frame_sec": pack_meta.get("frame_sec"),
            "rms_series": rms_series,
            "rms_stats": pack_meta.get("rms_stats"),
            "speech_segments": speech_segments,
            "pauses": pauses,
            "duration": pack_meta.get("duration"),
        }

    def _load_cached_face_pack(self, video_path: Path) -> Optional[Dict[str, object]]:
        if not self._is_cache_valid(video_path):
            return None
        if not self._is_artifact_complete("analysis_face"):
            return None
        face_path = self._artifact_path("analysis_face")
        if not face_path.exists():
            return None
        try:
            series = json.loads(face_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return {"series": series}

    def _write_analysis_meta(self, video_path: Path, srt_path: Path) -> None:
        meta = {
            "video_path": str(video_path),
            "video_mtime": video_path.stat().st_mtime,
            "video_size": video_path.stat().st_size,
            "srt_path": str(srt_path),
            "srt_mtime": srt_path.stat().st_mtime if srt_path.exists() else None,
            "srt_size": srt_path.stat().st_size if srt_path.exists() else None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._write_json(self._analysis_meta_path, meta)
        self._write_json(self._analysis_timestamp_path, {"timestamp": meta["timestamp"]})

    def _analysis_suffixes(self) -> List[str]:
        return [
            "analysis_audio",
            "analysis_audio_pack",
            "analysis_audio_rms",
            "analysis_audio_vad",
            "analysis_audio_pauses",
            "analysis_face",
            "analysis_clip_candidates_discovery",
            "analysis_clip_candidates",
            "analysis_clip_candidates_enriched",
            "analysis_llm_scores",
            "analysis_top_candidates",
            "analysis_cutlist",
            "analysis_meta",
            "analysis_timestamp",
        ]

    def _write_analysis_status(self) -> None:
        status = {}
        for suffix in self._analysis_suffixes():
            path = self._artifact_path(suffix) if suffix != "analysis_audio" else self._artifact_path(suffix, ext=".wav")
            status[suffix] = bool(path.exists())
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "artifacts": status,
        }
        self._write_json(self._analysis_status_path, payload)

    def _is_artifact_complete(self, suffix: str) -> bool:
        if not self._analysis_status_path.exists():
            return False
        try:
            status = json.loads(self._analysis_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        artifacts = status.get("artifacts", {})
        if not artifacts.get(suffix, False):
            return False
        path = self._artifact_path(suffix) if suffix != "analysis_audio" else self._artifact_path(suffix, ext=".wav")
        return path.exists()

    def _is_cache_valid(self, video_path: Path) -> bool:
        if not self._analysis_meta_path.exists():
            return False
        try:
            meta = json.loads(self._analysis_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if meta.get("video_path") != str(video_path):
            return False
        try:
            stat = video_path.stat()
        except OSError:
            return False
        return meta.get("video_mtime") == stat.st_mtime and meta.get("video_size") == stat.st_size

    def _blendshape_scores(self, results) -> Dict[str, float]:
        if not getattr(results, "face_blendshapes", None):
            return {}
        blendshapes = results.face_blendshapes[0]
        return {item.category_name: float(item.score) for item in blendshapes}

    def _blendshape_features(self, scores: Dict[str, float]) -> Dict[str, float]:
        smile = self._mean(
            [
                scores.get("mouthSmileLeft", 0.0),
                scores.get("mouthSmileRight", 0.0),
                scores.get("mouthSmile", 0.0),
            ]
        )
        brow_raise = self._mean(
            [
                scores.get("browInnerUp", 0.0),
                scores.get("browOuterUpLeft", 0.0),
                scores.get("browOuterUpRight", 0.0),
            ]
        )
        eye_blink = self._mean(
            [
                scores.get("eyeBlinkLeft", 0.0),
                scores.get("eyeBlinkRight", 0.0),
            ]
        )
        eye_open = max(0.0, 1.0 - eye_blink)
        mouth_open = scores.get("jawOpen", 0.0)
        return {
            "smile_index": float(smile),
            "brow_raise": float(brow_raise),
            "eye_open": float(eye_open),
            "mouth_open": float(mouth_open),
        }

    def _pose_from_matrix(self, results) -> Tuple[float, float, float]:
        matrices = getattr(results, "facial_transformation_matrixes", None)
        if not matrices:
            return 0.0, 0.0, 0.0
        try:
            matrix = np.array(matrices[0], dtype=np.float32)
            return self._matrix_to_euler(matrix)
        except Exception:
            return 0.0, 0.0, 0.0

    def _matrix_to_euler(self, matrix: np.ndarray) -> Tuple[float, float, float]:
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            return 0.0, 0.0, 0.0
        r = matrix[:3, :3]
        sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(r[2, 1], r[2, 2])
            yaw = math.atan2(-r[2, 0], sy)
            roll = math.atan2(r[1, 0], r[0, 0])
        else:
            pitch = math.atan2(-r[1, 2], r[1, 1])
            yaw = math.atan2(-r[2, 0], sy)
            roll = 0.0
        return (
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        )

    def _log_face_progress(
        self,
        processed: int,
        total_samples: Optional[int],
        last_bucket: int,
        last_log_time: float,
    ) -> tuple[int, float]:
        now = time.monotonic()
        if not total_samples:
            if now - last_log_time >= 10.0:
                self.logger.info("[SCENE] Face pack progress: %d frames", processed)
                return last_bucket, now
            return last_bucket, last_log_time
        percent = int((processed / total_samples) * 100)
        bucket = min(100, (percent // 5) * 5)
        if now - last_log_time >= 10.0 or bucket >= 100:
            self.logger.info(
                "[SCENE] Face pack progress: %d%% (%d/%d)",
                bucket,
                processed,
                total_samples,
            )
            return bucket, now
        return last_bucket, last_log_time

    def _generate_candidates(
        self,
        speech_segments: List[Dict[str, float]],
        pauses: List[Dict[str, float]],
        face_series: Optional[List[Dict[str, float]]],
        transcript_segments: Optional[List[SrtSegment]] = None,
    ) -> List[Dict[str, object]]:
        blocks = [Segment(**seg) for seg in speech_segments]
        transcript_candidates = self._generate_transcript_candidates(transcript_segments or [])
        pause_candidates = self._generate_pause_candidates(blocks)
        peak_candidates = self._generate_peak_candidates(pauses, face_series)
        candidates = transcript_candidates + pause_candidates + peak_candidates
        deduped = self._deduplicate_candidates(candidates)
        self.logger.info(
            "[SCENE] Candidate generation: pause=%d, peaks=%d, deduped=%d",
            len(pause_candidates),
            len(peak_candidates),
            len(deduped),
        )
        if transcript_candidates:
            self.logger.info(
                "[SCENE] Candidate generation: transcript=%d",
                len(transcript_candidates),
            )
        if not deduped:
            fallback = self._generate_fallback_candidates(blocks, source="fallback_blocks")
            if fallback:
                self.logger.info("[SCENE] Candidate fallback used: %d from speech blocks", len(fallback))
            deduped = fallback
        if not deduped and transcript_segments:
            transcript_blocks = [Segment(seg.start, seg.end) for seg in transcript_segments]
            fallback = self._generate_fallback_candidates(transcript_blocks, source="fallback_srt")
            if fallback:
                self.logger.info("[SCENE] Candidate fallback used: %d from transcript", len(fallback))
            deduped = fallback
        return self._assign_candidate_ids(deduped)

    def _generate_pause_candidates(self, blocks: List[Segment]) -> List[Dict[str, object]]:
        candidates = []
        for i in range(len(blocks)):
            block_duration = blocks[i].end - blocks[i].start
            if block_duration > self.config.max_clip_seconds:
                step = self.config.max_clip_seconds * 0.5
                window_start = blocks[i].start
                while window_start + self.config.min_clip_seconds <= blocks[i].end:
                    window_end = min(window_start + self.config.max_clip_seconds, blocks[i].end)
                    if window_end - window_start >= self.config.min_clip_seconds:
                        candidates.append(
                            {
                                "start": window_start,
                                "end": window_end,
                                "source": "long_block",
                            }
                        )
                    window_start += step
                continue
            for j in range(i, min(len(blocks), i + self.config.max_blocks_per_candidate)):
                start = blocks[i].start
                end = blocks[j].end
                duration = end - start
                if duration < self.config.min_clip_seconds:
                    continue
                if duration > self.config.max_clip_seconds:
                    break
                candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "source": "pause_blocks",
                    }
                )
        return candidates

    def _generate_transcript_candidates(
        self,
        segments: List[SrtSegment],
    ) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        if not segments:
            return candidates
        start_idx = 0
        while start_idx < len(segments):
            start_time = segments[start_idx].start
            end_idx = start_idx
            while end_idx < len(segments) and segments[end_idx].end - start_time < self.config.min_clip_seconds:
                end_idx += 1
            if end_idx >= len(segments):
                break
            end_idx_cursor = end_idx
            while end_idx_cursor < len(segments):
                end_time = segments[end_idx_cursor].end
                duration = end_time - start_time
                if duration > self.config.max_clip_seconds:
                    break
                if (not self.config.transcript_require_sentence_end) or self._ends_with_sentence(
                    segments,
                    start_idx,
                    end_idx_cursor,
                ):
                    candidates.append(
                        {
                            "start": start_time,
                            "end": end_time,
                            "source": "transcript",
                        }
                    )
                    if len(candidates) >= self.config.transcript_max_candidates:
                        return candidates
                end_idx_cursor += self.config.transcript_extend_step
            start_idx += self.config.transcript_start_step
        if not candidates and self.config.transcript_require_sentence_end and self.config.transcript_relax_sentence_end_if_empty:
            self.logger.info("[SCENE] Transcript candidates empty; relaxing sentence-end rule")
            relaxed = SceneDetectionConfig(
                **{**self.config.__dict__, "transcript_require_sentence_end": False}
            )
            original_config = self.config
            self.config = relaxed
            try:
                return self._generate_transcript_candidates(segments)
            finally:
                self.config = original_config
        return candidates

    def _generate_fallback_candidates(
        self,
        blocks: List[Segment],
        source: str,
        max_candidates: int = 200,
    ) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []
        if not blocks:
            return candidates
        for i in range(len(blocks)):
            start = blocks[i].start
            end = blocks[i].end
            j = i
            while j + 1 < len(blocks) and end - start < self.config.min_clip_seconds:
                j += 1
                end = blocks[j].end
            if end - start < self.config.min_clip_seconds:
                break
            max_end = start + self.config.max_clip_seconds
            if end > max_end:
                end = max_end
                for k in range(j, i, -1):
                    if blocks[k].end <= max_end:
                        end = blocks[k].end
                        break
            if end - start >= self.config.min_clip_seconds:
                candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "source": source,
                    }
                )
            if len(candidates) >= max_candidates:
                break
        return candidates

    def _normalize_candidate_list(self, candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
        normalized: List[Dict[str, object]] = []
        for idx, candidate in enumerate(candidates, start=1):
            start = float(candidate.get("start", candidate.get("start_time", 0.0)))
            end = float(candidate.get("end", candidate.get("end_time", 0.0)))
            if end <= start:
                continue
            candidate_id = candidate.get("candidate_id") or f"cand_{idx:03d}"
            normalized.append(
                {
                    "candidate_id": candidate_id,
                    "start": start,
                    "end": end,
                    "source": candidate.get("source", "llm_discovery"),
                }
            )
        return normalized

    def _ends_with_sentence(
        self,
        segments: List[SrtSegment],
        start_idx: int,
        end_idx: int,
    ) -> bool:
        if end_idx < 0 or end_idx >= len(segments):
            return False
        text = " ".join(seg.text for seg in segments[start_idx : end_idx + 1]).strip()
        if not text:
            return False
        text = re.sub(r"[\s\"'\\)\\]\\}]+$", "", text)
        return bool(re.search(r"[\\.!?…]$", text))

    def _generate_peak_candidates(
        self,
        pauses: List[Dict[str, float]],
        face_series: Optional[List[Dict[str, float]]],
    ) -> List[Dict[str, object]]:
        if not face_series:
            return []
        expressivity = np.array([item.get("expressivity_z", 0.0) for item in face_series], dtype=np.float32)
        if expressivity.size == 0:
            return []
        threshold = float(np.percentile(expressivity, self.config.peak_percentile))
        peaks = [
            item for item in face_series if float(item.get("expressivity_z", 0.0)) >= threshold
        ]
        if not peaks:
            return []

        pause_segments = [Segment(**pause) for pause in pauses]
        candidates = []
        for peak in peaks:
            t = float(peak["t"])
            start_pause = self._find_previous_pause_end(pause_segments, t)
            end_pause = self._find_next_pause_start(pause_segments, t)
            if start_pause is None or end_pause is None:
                continue
            duration = end_pause - start_pause
            if duration < self.config.min_clip_seconds or duration > self.config.max_clip_seconds:
                continue
            candidates.append(
                {
                    "start": start_pause,
                    "end": end_pause,
                    "source": "expressivity_peak",
                }
            )
        return candidates

    def _deduplicate_candidates(self, candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
        candidates = sorted(candidates, key=lambda item: (item["start"], item["end"]))
        result: List[Dict[str, object]] = []
        for candidate in candidates:
            keep = True
            for idx, existing in enumerate(result):
                if self._iou(candidate, existing) > self.config.dedup_iou:
                    keep = self._quality_hint(candidate) >= self._quality_hint(existing)
                    if keep:
                        result[idx] = candidate
                    break
            if keep and candidate not in result:
                result.append(candidate)
        return result

    def _assign_candidate_ids(self, candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
        assigned = []
        for idx, candidate in enumerate(candidates, start=1):
            assigned.append({**candidate, "candidate_id": f"cand_{idx:03d}"})
        return assigned

    def _enrich_candidates(
        self,
        candidates: List[Dict[str, object]],
        audio: Dict[str, object],
        transcript: Dict[str, object],
        face: Optional[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        segments = transcript["segments"]
        rms_series = audio["rms_series"]
        pauses = [Segment(**pause) for pause in audio["pauses"]]
        speech_segments = [Segment(**seg) for seg in audio["speech_segments"]]
        face_series = face["series"] if face else None

        enriched = []
        for candidate in candidates:
            start = float(candidate["start"])
            end = float(candidate["end"])
            duration = max(end - start, 0.01)
            transcript_excerpt = self._extract_transcript_excerpt(segments, start, end)
            word_count = len(transcript_excerpt.split())
            words_per_second = word_count / duration if duration > 0 else 0.0

            audio_stats = self._build_audio_stats(
                rms_series,
                speech_segments,
                pauses,
                start,
                end,
            )
            face_stats = self._build_face_stats(face_series, start, end)
            whisper_confidence = self._build_whisper_confidence(transcript.get("transcript_json"), start, end)

            enriched.append(
                {
                    **candidate,
                    "duration_seconds": duration,
                    "transcript_excerpt": transcript_excerpt,
                    "words_per_second": round(words_per_second, 3),
                    "audio_stats": audio_stats,
                    "face_stats": face_stats,
                    "whisper_confidence": whisper_confidence,
                }
            )
        return enriched

    def _apply_llm_scores(
        self,
        candidates: List[Dict[str, object]],
        llm_scores: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        score_map = {item.get("candidate_id"): item for item in llm_scores}
        updated = []
        for candidate in candidates:
            score_info = score_map.get(candidate["candidate_id"])
            if not score_info:
                continue
            updated.append({**candidate, **score_info})
        return updated

    def _rank_candidates(self, candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not candidates:
            return []

        audio_scores = [self._audio_score_inputs(cand) for cand in candidates]
        face_scores = [self._face_score_inputs(cand) for cand in candidates]

        rms_percentiles = [item[0] for item in audio_scores if item[0] is not None]
        rms_variances = [item[1] for item in audio_scores if item[1] is not None]
        speech_ratios = [item[2] for item in audio_scores if item[2] is not None]
        rms_rise_rates = [item[3] for item in audio_scores if item[3] is not None]
        rms_peak_rates = [item[4] for item in audio_scores if item[4] is not None]

        expressivity_peaks = [item[0] for item in face_scores if item[0] is not None]
        expressivity_vars = [item[1] for item in face_scores if item[1] is not None]

        ranked = []
        filtered_counts = {
            "speech_ratio": 0,
            "words_per_second": 0,
            "whisper_conf": 0,
            "incomplete_thought": 0,
        }
        for candidate, audio_tuple, face_tuple in zip(candidates, audio_scores, face_scores):
            if not self._passes_filters(candidate, filtered_counts):
                continue
            content_score = float(candidate.get("content_score", 0))
            s_text = content_score / 10.0
            s_cut = 1.0 if self._pause_quality(candidate) else 0.0

            s_audio = self._mean(
                [
                    self._normalize(audio_tuple[0], rms_percentiles),
                    self._normalize(audio_tuple[1], rms_variances),
                    self._normalize(audio_tuple[2], speech_ratios),
                    self._normalize(audio_tuple[3], rms_rise_rates),
                    self._normalize(audio_tuple[4], rms_peak_rates),
                ]
            )

            s_face = None
            if face_tuple[0] is not None and face_tuple[1] is not None:
                s_face = self._mean(
                    [
                        self._normalize(face_tuple[0], expressivity_peaks),
                        self._normalize(face_tuple[1], expressivity_vars),
                    ]
                )

            score = self._weighted_score(s_text, s_cut, s_audio, s_face)
            if candidate.get("complete_thought") is False:
                score = max(0.0, score - self.config.incomplete_thought_penalty)
            ranked.append({**candidate, "mixed_score": round(score, 4)})

        ranked.sort(key=lambda item: item["mixed_score"], reverse=True)
        self.logger.info(
            "[SCENE] Ranking: input=%d, passed_filters=%d",
            len(candidates),
            len(ranked),
        )
        if any(filtered_counts.values()):
            self.logger.info(
                "[SCENE] Ranking filters: speech_ratio=%d, words_per_second=%d, whisper_conf=%d, incomplete_thought=%d",
                filtered_counts["speech_ratio"],
                filtered_counts["words_per_second"],
                filtered_counts["whisper_conf"],
                filtered_counts["incomplete_thought"],
            )
        return ranked

    def _apply_diversity(self, candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
        seen = {}
        for candidate in candidates:
            label = (candidate.get("topic_label") or "").strip().lower()
            if not label:
                label = candidate["candidate_id"]
            if label not in seen:
                seen[label] = candidate
        return list(seen.values())

    def _passes_filters(self, candidate: Dict[str, object], counts: Dict[str, int]) -> bool:
        if candidate.get("complete_thought") is False:
            counts["incomplete_thought"] += 1
        audio_stats = candidate.get("audio_stats") or {}
        speech_ratio = audio_stats.get("speech_ratio", 1.0)
        if speech_ratio < self.config.speech_ratio_min:
            words_per_second = float(candidate.get("words_per_second") or 0.0)
            if words_per_second < self.config.words_per_second_min:
                counts["speech_ratio"] += 1
                counts["words_per_second"] += 1
                return False
        whisper_conf = candidate.get("whisper_confidence") or {}
        if whisper_conf and whisper_conf.get("bad_segment_ratio", 0.0) > self.config.whisper_bad_ratio_max:
            counts["whisper_conf"] += 1
            return False
        return True

    def _pause_quality(self, candidate: Dict[str, object]) -> bool:
        audio_stats = candidate.get("audio_stats") or {}
        return bool(audio_stats.get("pause_quality_start")) and bool(audio_stats.get("pause_quality_end"))

    def _refine_boundaries(
        self,
        start: float,
        end: float,
        pauses: List[Segment],
        alignment: Optional[Dict[str, object]],
    ) -> Tuple[float, float, Dict[str, object]]:
        meta = {"snapped_by": None, "editorial_offsets": None}

        if alignment:
            first_word = alignment.get("first_word_start")
            last_word = alignment.get("last_word_end")
            if first_word is not None:
                start = max(0.0, float(first_word) - self.config.start_offset_seconds)
            if last_word is not None:
                end = max(start, float(last_word) + self.config.end_offset_seconds)
            meta["editorial_offsets"] = {
                "start_offset": self.config.start_offset_seconds,
                "end_offset": self.config.end_offset_seconds,
            }
        else:
            start = max(0.0, start - self.config.start_offset_seconds_no_align)
            end = max(start, end + self.config.end_offset_seconds_no_align)
            meta["editorial_offsets"] = {
                "start_offset": self.config.start_offset_seconds_no_align,
                "end_offset": self.config.end_offset_seconds_no_align,
            }

        snapped_start = self._snap_start_to_pause(start, pauses)
        snapped_end = self._snap_end_to_pause(end, pauses)
        if snapped_start is not None:
            start = snapped_start
            meta["snapped_by"] = "start_pause"
        if snapped_end is not None and snapped_end > start:
            end = snapped_end
            meta["snapped_by"] = "end_pause" if meta["snapped_by"] is None else "start_end_pause"
        return start, end, meta

    def _whisperx_align_candidate(
        self,
        video_path: Path,
        candidate_id: str,
        start: float,
        end: float,
        transcript_segments: List[SrtSegment],
    ) -> Optional[Dict[str, object]]:
        if not self.config.whisperx_enabled:
            return None
        if self.config.whisperx_env_name:
            return self._whisperx_align_candidate_external(
                video_path,
                candidate_id,
                start,
                end,
                transcript_segments,
            )
        try:
            import torch  # type: ignore
            try:
                import transformers  # type: ignore
                if not hasattr(transformers, "Pipeline"):
                    from transformers.pipelines import Pipeline  # type: ignore
                    transformers.Pipeline = Pipeline  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                from omegaconf import DictConfig  # type: ignore
                from omegaconf.base import ContainerMetadata  # type: ignore
                from omegaconf.listconfig import ListConfig  # type: ignore
                torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])
            except Exception:
                pass
            if not hasattr(torch, "_ai_videoclipper_load_patched"):
                original_load = torch.load

                def _load_compat(*args, **kwargs):  # type: ignore[override]
                    kwargs.setdefault("weights_only", False)
                    return original_load(*args, **kwargs)

                torch.load = _load_compat  # type: ignore[assignment]
                torch._ai_videoclipper_load_patched = True  # type: ignore[attr-defined]
            import whisperx  # type: ignore
        except Exception:
            self.logger.info("[SCENE] WhisperX alignment skipped (whisperx not installed)")
            return None

        pad = self.config.whisperx_pad_seconds
        window_start = max(0.0, start - pad)
        window_end = max(window_start + 0.5, end + pad)
        window_path = self._artifact_path(f"analysis_alignment_{candidate_id}", ext=".wav")

        self._extract_audio_window(video_path, window_path, window_start, window_end)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        try:
            self.logger.info("[SCENE] WhisperX aligning %s (%.2fs-%.2fs)", candidate_id, start, end)
            audio = whisperx.load_audio(str(window_path))
            model = self._get_whisperx_model(whisperx, device, compute_type)
            result = model.transcribe(audio)
            if not result.get("segments"):
                return None
            align_segments = self._segments_from_srt(
                transcript_segments,
                window_start,
                window_end,
            )
            if align_segments:
                self.logger.info(
                    "[SCENE] WhisperX alignment using SRT segments (%d)",
                    len(align_segments),
                )
            else:
                align_segments = result["segments"]
                self.logger.info("[SCENE] WhisperX alignment using WhisperX segments")
            align_model, metadata = self._get_whisperx_align_model(
                whisperx,
                result["language"],
                device,
            )
            aligned = whisperx.align(
                align_segments,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            words = []
            for seg in aligned.get("segments", []):
                words.extend(seg.get("words", []))
            if not words:
                return None
            first_word = min(words, key=lambda w: w["start"])
            last_word = max(words, key=lambda w: w["end"])
            self.logger.info(
                "[SCENE] WhisperX aligned words=%d (first=%.2fs last=%.2fs)",
                len(words),
                window_start + float(first_word["start"]),
                window_start + float(last_word["end"]),
            )
            return {
                "first_word_start": window_start + float(first_word["start"]),
                "last_word_end": window_start + float(last_word["end"]),
            }
        except Exception as exc:
            self.logger.warning(f"[SCENE] WhisperX alignment failed: {exc}")
            return None

    def _whisperx_align_candidate_external(
        self,
        video_path: Path,
        candidate_id: str,
        start: float,
        end: float,
        transcript_segments: List[SrtSegment],
    ) -> Optional[Dict[str, object]]:
        env_name = self.config.whisperx_env_name
        if not env_name:
            return None
        pad = self.config.whisperx_pad_seconds
        window_start = max(0.0, start - pad)
        window_end = max(window_start + 0.5, end + pad)
        window_path = self.artifacts_dir / f"align_{candidate_id}.wav"
        output_path = self._artifact_path(f"analysis_alignment_{candidate_id}")

        self._extract_audio_window(video_path, window_path, window_start, window_end)
        align_segments = self._segments_from_srt(
            transcript_segments,
            window_start,
            window_end,
        )
        payload = {
            "window_start": window_start,
            "window_end": window_end,
            "segments": align_segments,
        }
        segments_path = self._artifact_path(f"analysis_alignment_{candidate_id}_segments")
        self._write_json(segments_path, payload)

        script_path = Path(__file__).resolve().parent / "whisperx_align.py"
        if not script_path.exists():
            raise RuntimeError(f"WhisperX alignment script missing: {script_path}")

        # Use GPU - fail with clear error if it doesn't work
        # Set LD_LIBRARY_PATH to include all NVIDIA CUDA libraries from conda environment
        env_nvidia_libs = f"~/miniconda3/envs/{env_name}/lib/python3.10/site-packages/nvidia/*/lib"
        cmd = [
            "bash",
            "-lc",
            "export AI_VIDECLIPPER_FORCE_CPU=0 && "
            f"for dir in {env_nvidia_libs}; do export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH; done && "
            "source ~/miniconda3/etc/profile.d/conda.sh "
            f"&& conda activate {env_name} "
            f'&& python "{script_path}" '
            f'--audio "{window_path}" '
            f'--segments "{segments_path}" '
            f'--output "{output_path}"',
        ]

        self.logger.info(
            "[SCENE] WhisperX aligning [GPU] (env=%s) %s (%.2fs-%.2fs)",
            env_name,
            candidate_id,
            start,
            end,
        )
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            error_msg = f"WhisperX external alignment failed [GPU]: {message}"
            if self.config.whisperx_require_env:
                raise RuntimeError(error_msg)
            self.logger.warning(f"[SCENE] {error_msg}")
            return None
        try:
            alignment = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            error_msg = f"WhisperX alignment output invalid: {exc}"
            if self.config.whisperx_require_env:
                raise RuntimeError(error_msg)
            self.logger.warning(f"[SCENE] {error_msg}")
            return None
        if alignment.get("first_word_start") is None or alignment.get("last_word_end") is None:
            error_msg = "WhisperX alignment output missing word boundaries"
            if self.config.whisperx_require_env:
                raise RuntimeError(error_msg)
            self.logger.warning(f"[SCENE] {error_msg}")
            return None
        alignment["first_word_start"] = window_start + float(alignment["first_word_start"])
        alignment["last_word_end"] = window_start + float(alignment["last_word_end"])
        self.logger.info(
            "[SCENE] WhisperX aligned words=%d (first=%.2fs last=%.2fs)",
            alignment.get("word_count", 0),
            alignment["first_word_start"],
            alignment["last_word_end"],
        )
        return alignment

    def _compute_key_moment(
        self,
        start: float,
        end: float,
        face_series: Optional[List[Dict[str, float]]],
        rms_series: List[Dict[str, float]],
    ) -> Optional[Dict[str, object]]:
        duration = end - start
        if duration < 60.0:
            return None
        if not face_series:
            return None

        face_window = [item for item in face_series if start <= item["t"] <= end]
        rms_window = [item for item in rms_series if start <= item["t"] <= end]
        if not face_window or not rms_window:
            return None

        rms_values = np.array([item["rms"] for item in rms_window], dtype=np.float32)
        rms_mean = float(rms_values.mean()) if rms_values.size else 0.0
        rms_std = float(rms_values.std()) if rms_values.size else 1.0
        rms_std = rms_std if rms_std > 1e-6 else 1.0

        window_len = min(10.0, duration)
        best = None
        best_score = -1.0

        for item in face_window:
            t = float(item["t"])
            window_start = max(start, t - window_len / 2.0)
            window_end = window_start + window_len
            if window_end > end:
                window_end = end
                window_start = max(start, window_end - window_len)

            window_face = [f for f in face_window if window_start <= f["t"] <= window_end]
            window_rms = [r for r in rms_window if window_start <= r["t"] <= window_end]
            if not window_face or not window_rms:
                continue

            face_mean = float(np.mean([f.get("expressivity_z", 0.0) for f in window_face]))
            rms_mean_window = float(np.mean([r["rms"] for r in window_rms]))
            rms_z = (rms_mean_window - rms_mean) / rms_std
            intensity = abs(rms_z)

            if face_mean < 0.8 or intensity < 0.6:
                continue

            score = face_mean + intensity
            if score > best_score:
                best_score = score
                best = {
                    "start": round(window_start, 3),
                    "end": round(window_end, 3),
                    "face_mean_z": round(face_mean, 3),
                    "rms_z": round(rms_z, 3),
                    "score": round(score, 3),
                }

        if best:
            self.logger.info(
                "[SCENE] Key moment: %.2fs-%.2fs (score=%.2f)",
                best["start"],
                best["end"],
                best["score"],
            )
        return best

    def _segments_from_srt(
        self,
        segments: List[SrtSegment],
        window_start: float,
        window_end: float,
    ) -> List[Dict[str, object]]:
        aligned: List[Dict[str, object]] = []
        for seg in segments:
            if seg.end <= window_start or seg.start >= window_end:
                continue
            start = max(0.0, seg.start - window_start)
            end = max(start + 0.01, seg.end - window_start)
            text = seg.text.strip()
            if not text:
                continue
            aligned.append({"start": start, "end": end, "text": text})
        return aligned

    def _get_whisperx_model(self, whisperx_module, device: str, compute_type: str):
        if self._whisperx_model is None or self._whisperx_device != device:
            self._whisperx_model = whisperx_module.load_model(
                "large-v2",
                device=device,
                compute_type=compute_type,
            )
            self._whisperx_device = device
        return self._whisperx_model

    def _get_whisperx_align_model(self, whisperx_module, language: str, device: str):
        if (
            self._whisperx_align_model is None
            or self._whisperx_language != language
            or self._whisperx_device != device
        ):
            self._whisperx_align_model, self._whisperx_metadata = whisperx_module.load_align_model(
                language_code=language,
                device=device,
            )
            self._whisperx_language = language
        return self._whisperx_align_model, self._whisperx_metadata

    def _extract_transcript_excerpt(
        self,
        segments: List[SrtSegment],
        start: float,
        end: float,
    ) -> str:
        texts = [seg.text for seg in segments if seg.end > start and seg.start < end]
        return " ".join(texts).strip()

    def _build_audio_stats(
        self,
        rms_series: List[Dict[str, float]],
        speech_segments: List[Segment],
        pauses: List[Segment],
        start: float,
        end: float,
    ) -> Dict[str, object]:
        rms_values = [item["rms"] for item in rms_series if start <= item["t"] <= end]
        rms_values_np = np.array(rms_values, dtype=np.float32) if rms_values else np.array([], dtype=np.float32)
        rms_diff = np.diff(rms_values_np) if rms_values_np.size > 1 else np.array([], dtype=np.float32)
        rms_rise_rate = float(np.mean(rms_diff[rms_diff > 0])) if rms_diff.size else 0.0
        if rms_values_np.size:
            peak_threshold = float(np.percentile(rms_values_np, 85))
            peak_rate = float(np.mean(rms_values_np >= peak_threshold))
        else:
            peak_rate = 0.0
        speech_ratio = self._speech_ratio(speech_segments, start, end)

        return {
            "pause_quality_start": self._pause_near(start, pauses),
            "pause_quality_end": self._pause_near(end, pauses),
            "speech_ratio": round(speech_ratio, 3),
            "rms_mean": float(rms_values_np.mean()) if rms_values_np.size else 0.0,
            "rms_peak": float(rms_values_np.max()) if rms_values_np.size else 0.0,
            "rms_percentile": float(np.percentile(rms_values_np, 90)) if rms_values_np.size else 0.0,
            "rms_variance": float(rms_values_np.var()) if rms_values_np.size else 0.0,
            "rms_rise_rate": rms_rise_rate,
            "rms_peak_rate": peak_rate,
        }

    def _build_face_stats(
        self,
        face_series: Optional[List[Dict[str, float]]],
        start: float,
        end: float,
    ) -> Optional[Dict[str, object]]:
        if not face_series:
            return None
        window = [item for item in face_series if start <= item["t"] <= end]
        if not window:
            return None
        face_ok = [item.get("face_ok", 0) for item in window]
        expressivity = [item.get("expressivity_z", 0.0) for item in window]
        face_ok_ratio = sum(face_ok) / len(face_ok) if face_ok else 0.0
        expressivity_np = np.array(expressivity, dtype=np.float32) if expressivity else np.array([], dtype=np.float32)
        return {
            "face_ok_ratio": round(face_ok_ratio, 3),
            "expressivity_peak": float(expressivity_np.max()) if expressivity_np.size else 0.0,
            "expressivity_variance": float(expressivity_np.var()) if expressivity_np.size else 0.0,
        }

    def _build_whisper_confidence(
        self,
        transcript_json: Optional[Dict[str, object]],
        start: float,
        end: float,
    ) -> Optional[Dict[str, object]]:
        if not transcript_json:
            return None
        segments = transcript_json.get("segments", [])
        relevant = [
            seg for seg in segments if seg.get("end", 0) > start and seg.get("start", 0) < end
        ]
        if not relevant:
            return None
        logprobs = [seg.get("avg_logprob") for seg in relevant if seg.get("avg_logprob") is not None]
        no_speech = [seg.get("no_speech_prob") for seg in relevant if seg.get("no_speech_prob") is not None]
        bad_segments = [
            seg for seg in relevant if seg.get("no_speech_prob", 0) > 0.6 or seg.get("avg_logprob", 0) < -1.0
        ]
        return {
            "avg_logprob_mean": float(np.mean(logprobs)) if logprobs else None,
            "no_speech_prob_mean": float(np.mean(no_speech)) if no_speech else None,
            "bad_segment_ratio": len(bad_segments) / len(relevant) if relevant else 0.0,
        }

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.config.sample_rate),
            "-f",
            "wav",
            str(audio_path),
        ]
        self._run_cmd(cmd, "audio extraction")

    def _extract_audio_window(
        self,
        video_path: Path,
        audio_path: Path,
        start: float,
        end: float,
    ) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.config.sample_rate),
            "-f",
            "wav",
            str(audio_path),
        ]
        self._run_cmd(cmd, "audio window extraction")

    def _run_cmd(self, cmd: List[str], label: str) -> None:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"{label} failed: ffmpeg not found") from exc
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr.strip() if exc.stderr else str(exc)
            raise RuntimeError(f"{label} failed: {msg}") from exc

    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        with wave.open(str(audio_path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if sample_width == 2:
            dtype = np.int16
            max_value = 32768.0
        elif sample_width == 4:
            dtype = np.int32
            max_value = 2147483648.0
        else:
            raise RuntimeError(f"Unsupported sample width: {sample_width}")
        audio = np.frombuffer(frames, dtype=dtype).astype(np.float32) / max_value
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, sample_rate

    def _compute_rms(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_sec: float,
    ) -> List[Dict[str, float]]:
        frame_len = max(int(sample_rate * frame_sec), 1)
        count = len(audio) // frame_len
        if count <= 0:
            return []
        trimmed = audio[: count * frame_len]
        frames = trimmed.reshape(count, frame_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        times = np.arange(count) * frame_len / sample_rate
        return [{"t": float(t), "rms": float(val)} for t, val in zip(times, rms)]

    def _compute_vad_segments(
        self,
        rms_series: List[Dict[str, float]],
        frame_sec: float,
        stats: Dict[str, float],
    ) -> List[Segment]:
        if not rms_series:
            return []
        threshold = stats.get("p60", stats.get("p50", 0.0))
        speech_flags = [item["rms"] >= threshold for item in rms_series]
        speech_flags = self._fill_short_gaps(speech_flags, frame_sec)
        if speech_flags:
            speech_ratio = sum(1 for flag in speech_flags if flag) / len(speech_flags)
            self.logger.info(
                "[SCENE] VAD threshold=%.6f, speech_frames_ratio=%.3f",
                threshold,
                speech_ratio,
            )

        segments: List[Segment] = []
        start = None
        for idx, is_speech in enumerate(speech_flags):
            t = idx * frame_sec
            if is_speech and start is None:
                start = t
            if not is_speech and start is not None:
                end = t
                if end - start >= self.config.vad_min_speech:
                    segments.append(Segment(start, end))
                start = None
        if start is not None:
            end = len(speech_flags) * frame_sec
            if end - start >= self.config.vad_min_speech:
                segments.append(Segment(start, end))

        merged: List[Segment] = []
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue
            gap = seg.start - merged[-1].end
            if gap <= self.config.vad_merge_gap:
                merged[-1].end = seg.end
            else:
                merged.append(seg)
        return merged

    def _fill_short_gaps(self, flags: List[bool], frame_sec: float) -> List[bool]:
        if not flags:
            return flags
        gap_frames = int(round((self.config.vad_gap_fill_ms / 1000.0) / frame_sec))
        gap_frames = max(gap_frames, 1)
        filled = flags[:]
        i = 0
        while i < len(filled):
            if filled[i]:
                i += 1
                continue
            start = i
            while i < len(filled) and not filled[i]:
                i += 1
            end = i
            run_length = end - start
            has_left = start > 0 and filled[start - 1]
            has_right = end < len(filled) and filled[end]
            if run_length <= gap_frames and has_left and has_right:
                for j in range(start, end):
                    filled[j] = True
        return filled

    def _derive_pauses(self, speech_segments: List[Segment], duration: float) -> List[Segment]:
        pauses: List[Segment] = []
        last_end = 0.0
        for seg in speech_segments:
            if seg.start - last_end >= self.config.vad_pause_min:
                pauses.append(Segment(last_end, seg.start))
            last_end = seg.end
        if duration - last_end >= self.config.vad_pause_min:
            pauses.append(Segment(last_end, duration))
        return pauses

    def _audio_duration(self, audio: np.ndarray, sample_rate: int) -> float:
        if sample_rate <= 0:
            return 0.0
        return len(audio) / sample_rate

    def _speech_ratio(self, segments: List[Segment], start: float, end: float) -> float:
        total = 0.0
        for seg in segments:
            overlap = max(0.0, min(end, seg.end) - max(start, seg.start))
            total += overlap
        duration = max(end - start, 0.01)
        return total / duration

    def _pause_near(self, time_value: float, pauses: List[Segment]) -> bool:
        for pause in pauses:
            if pause.start <= time_value <= pause.end:
                return True
            if abs(pause.start - time_value) <= 0.2:
                return True
            if abs(pause.end - time_value) <= 0.2:
                return True
        return False

    def _find_previous_pause_end(self, pauses: List[Segment], t: float) -> Optional[float]:
        ends = [pause.end for pause in pauses if pause.end <= t]
        return max(ends) if ends else None

    def _find_next_pause_start(self, pauses: List[Segment], t: float) -> Optional[float]:
        starts = [pause.start for pause in pauses if pause.start >= t]
        return min(starts) if starts else None

    def _snap_start_to_pause(self, time_value: float, pauses: List[Segment]) -> Optional[float]:
        window = self.config.snap_window_seconds
        before = [pause.end for pause in pauses if pause.end <= time_value and time_value - pause.end <= window]
        after = [pause.end for pause in pauses if pause.end >= time_value and pause.end - time_value <= window]
        if before:
            return max(before)
        if after:
            return min(after)
        return None

    def _snap_end_to_pause(self, time_value: float, pauses: List[Segment]) -> Optional[float]:
        window = self.config.snap_window_seconds
        after = [pause.start for pause in pauses if pause.start >= time_value and pause.start - time_value <= window]
        before = [pause.start for pause in pauses if pause.start <= time_value and time_value - pause.start <= window]
        if after:
            return min(after)
        if before:
            return max(before)
        return None

    def _quality_hint(self, candidate: Dict[str, object]) -> float:
        duration = float(candidate["end"]) - float(candidate["start"])
        target = (self.config.min_clip_seconds + self.config.max_clip_seconds) / 2.0
        return 1.0 - min(abs(duration - target) / target, 1.0)

    def _iou(self, a: Dict[str, object], b: Dict[str, object]) -> float:
        start = max(float(a["start"]), float(b["start"]))
        end = min(float(a["end"]), float(b["end"]))
        intersection = max(0.0, end - start)
        union = max(float(a["end"]), float(b["end"])) - min(float(a["start"]), float(b["start"]))
        if union <= 0:
            return 0.0
        return intersection / union

    def _audio_score_inputs(
        self,
        candidate: Dict[str, object],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        stats = candidate.get("audio_stats") or {}
        return (
            stats.get("rms_percentile"),
            stats.get("rms_variance"),
            stats.get("speech_ratio"),
            stats.get("rms_rise_rate"),
            stats.get("rms_peak_rate"),
        )

    def _face_score_inputs(self, candidate: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
        stats = candidate.get("face_stats") or {}
        return (
            stats.get("expressivity_peak"),
            stats.get("expressivity_variance"),
        )

    def _normalize(self, value: Optional[float], population: List[float]) -> float:
        if value is None or not population:
            return 0.0
        min_v = min(population)
        max_v = max(population)
        if math.isclose(min_v, max_v):
            return 0.5
        return (value - min_v) / (max_v - min_v)

    def _mean(self, values: Iterable[float]) -> float:
        vals = [val for val in values if val is not None]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def _weighted_score(self, s_text: float, s_cut: float, s_audio: float, s_face: Optional[float]) -> float:
        weights = {
            "text": self.config.weight_text,
            "cut": self.config.weight_cut,
            "audio": self.config.weight_audio,
            "face": self.config.weight_face,
        }
        score_parts = {
            "text": s_text,
            "cut": s_cut,
            "audio": s_audio,
            "face": s_face,
        }
        active = {k: v for k, v in score_parts.items() if v is not None}
        weight_sum = sum(weights[k] for k in active.keys())
        if weight_sum <= 0:
            return 0.0
        return sum(weights[k] * active[k] for k in active.keys()) / weight_sum

    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
