    from types import SimpleNamespace

    from atrain_transcriber import ATrainTranscriber


    def test_transcribe_finds_srt(tmp_path) -> None:
        conda_sh = tmp_path / "conda.sh"
        conda_sh.write_text("# stub")
        output_dir = tmp_path / "transcriptions"
        output_dir.mkdir()
        source = tmp_path / "video.mp4"
        source.write_text("stub")

        srt_content = "1
00:00:00,000 --> 00:00:01,000
Hi

"

        def runner(cmd, capture_output=True, text=True):
            srt_path = output_dir / "video.srt"
            srt_path.write_text(srt_content)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        transcriber = ATrainTranscriber(
            conda_sh=str(conda_sh),
            output_dir=str(output_dir),
            runner=runner,
        )
        result = transcriber.transcribe(str(source))
        assert result.srt_path.exists()
        assert result.segments
