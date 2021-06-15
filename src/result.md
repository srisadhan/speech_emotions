
# Speech Emotion Recognition

## Two models are trained to estimate emotion, both are autoencoders with the 256 latent embeddings. This process is also performed for spectrograms with y-axis as **power** and also with **dB**
 - AE directly trained on the speech statement 1 from RAVDESS (**without_speaker tag**)

 - AE trained with speaker embeddings from speaker encoder/voice encoder (**with_speaker tag**)
   - Latent space conditioning is applied by multiplying the Emotion and Speaker embeddings elementwise and used for decoder training.
   - Classification is performed using the Emotion embeddings 


## Models are trained on speech 1 data and tested on 
- Statement 2 data with same actors involved in training
- Statement 1 and 2 data of two actors (1 Male and 1 Female) not included in training
 

# EmotionNet_AE: 8 mels using STFT along with speaker embeddings perform better than 40 mels w/o speaker embedding and 80 mels with and w/o speaker embeddings