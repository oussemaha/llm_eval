from omnivoice import OmniVoice
import soundfile as sf
import torch



class TTS:
    def __init__(self):
        self.model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype=torch.float16
        )

    def generate(self, text: str) -> np.ndarray:
        audio = self.model.generate(
            text=text,
        )
        #sf.write("out_fr.wav", audio[0], 24000)
        return audio
