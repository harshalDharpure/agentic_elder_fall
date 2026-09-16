# Vision Pilot Status

- **UR Fall download:** 28/70 zips (running in background)
- **Pilot waiter:** waiting for download DONE, then runs CNN → zero-shot VLM → LoRA
- **GPU:** A100 80GB
- **Model:** Qwen2-VL-2B-Instruct (prefetching)

## Watch logs
```bash
tail -f logs/urfall_download.log
tail -f logs/vision_pilot.log
tail -f logs/vision_pilot_inner.log
```

## Verdict when done
`results/vision/PILOT_VERDICT.md` and `results/vision/pilot_go_nogo.json`
