# Recording the SahaayakAI demo video

This guide produces a single-take screen recording that hits every spec
deliverable. Total runtime: ~3 minutes.

## One-time setup (do this BEFORE you start recording)

1. Open **Windows Terminal** (not the legacy `cmd.exe` — colors won't render there).
2. `cd` to the project root:
   ```
   cd "C:\Users\kansa\Desktop\Projects\Hackathons\AI For Future Workforce (Intel)\sahaayak_ai"
   ```
3. Make sure the venv is active and dependencies are installed.
4. Pre-warm the Phi-3 NPU compile by running the simplifier once:
   ```
   $env:PYTHONPATH="src"; python -m sahaayak.main --simplify demo/sample_email.txt > $null
   ```
   The first run takes ~4 minutes (NPU compile cache is empty); subsequent
   runs reuse the cache and complete in ~5–10 s. **Always pre-warm** before
   recording, or your video will have a 4-minute dead air.
5. Position the Windows Terminal at ~1280x800 in the centre of your screen
   so OBS / your screen recorder can capture both the terminal and the
   cursor freely.

## Start the recording

Begin your screen recorder. Then in the terminal:

```
$env:PYTHONPATH="src"
python scripts/demo_full.py
```

The script walks through 4 phases and pauses between each. Hit ENTER
when you want to advance. Suggested narration:

### Phase 1 — Device detection (~10 s)
> "SahaayakAI starts by enumerating the Intel devices available on this
> Core Ultra 7 258V — the OpenVINO Runtime sees NPU, Arc 140V iGPU, and CPU.
> Vision and LLM both route to the NPU."

### Phase 2 — Real benchmark numbers (~15 s)
> "These are real per-device numbers from this laptop, not theoretical.
> The iris pipeline runs at 244 FPS on CPU, 177 FPS on NPU. Both blow past
> the 30 FPS budget for live tracking."

### Phase 3 — Live cursor + winks (~45 s)
> "Now I face the camera. MediaPipe FaceMesh tracks 478 facial landmarks.
> My head moves the cursor. I wink with my left eye — left click. Right eye
> — right click. F12 is a hard kill switch; pushing the mouse into a screen
> corner is a fail-safe."
>
> Demo it: move cursor across the screen. Wink left. Wink right. Both
> eyes blinking together does NOT click — only intentional winks fire.

### Phase 4 — Phi-3 email co-pilot (~10 s, post-warmup)
> "A 240-word email gets summarised by Phi-3-mini-int4 running entirely on
> the Intel NPU — zero cloud calls. TL;DR, key points, action items with
> deadlines, and three reply drafts. Each draft carries the AI-generated
> disclaimer."

End with the closing banner. Stop recording.

## Useful flags during rehearsal

```
python scripts/demo_full.py --no-pause                   # auto-advance
python scripts/demo_full.py --skip-llm                   # skip Phase 4
python scripts/demo_full.py --skip-cursor                # skip Phase 3
python scripts/demo_full.py --cursor-seconds 30          # shorter cursor demo
```

## Things to verify on playback

- [ ] Phase 1 banner shows NPU + GPU + CPU all detected.
- [ ] Phase 2 shows real numbers (not zeros).
- [ ] Cursor visibly moves with your head in Phase 3.
- [ ] At least one LEFT click and one RIGHT click fire from winks.
- [ ] Phase 4 shows the TL;DR, action items, and 3 reply drafts.
- [ ] No filenames containing webcam frames appear in the project tree
  (the privacy invariant is the entire point).

## File locations referenced in the video

- Source: `sahaayak_ai/src/sahaayak/`
- Docs:  `sahaayak_ai/docs/{ARCHITECTURE,RESPONSIBLE_AI,INTEL_STACK_RATIONALE}.md`
- Benchmark JSON: `sahaayak_ai/benchmark_report.json`
- Sample email used in Phase 4: `sahaayak_ai/demo/sample_email.txt`
