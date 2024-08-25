# Just a setting I quite liked:
# This colab: StyleTTS2_Demo_LibriTTS.ipynb

reference_dicts = {}
reference_dicts['suggestions'] = "Demo/reference_audio/suggestions.wav" # Took this from their demo page an renamed. 
reference_dicts['1789_142896'] = "Demo/reference_audio/1789_142896_000022_000005.wav"

noise = torch.randn(1,1,256).to(device)
for k, path in reference_dicts.items():
    ref_s = compute_style(path)
    start = time.time()
    wav = inference(text, ref_s, alpha=0.9, beta=0.7, diffusion_steps=5, embedding_scale=1) # didn't play much, but alpha=0.9 made it go from good to really good.
    rtf = (time.time() - start) / (len(wav) / 24000)
    print(f"RTF = {rtf:5f}")
    import IPython.display as ipd
    print(k + ' Synthesized:')
    display(ipd.Audio(wav, rate=24000, normalize=False))
    print('Reference:')
    display(ipd.Audio(path, rate=24000, normalize=False))