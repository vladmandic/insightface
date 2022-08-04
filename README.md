# insightface
TFJS port of InsightFace
# InsightFace for TFJS

Demo uses `Human` library to detect and crop faces before running `InsightFace` models  
as well as using optimized distance methods present in `Human` to measure face similarities  
(you can use any other face detection method)  
## Models

Repository contains pretrained **TFJS graph models** for the following **InsightFace** variations  
Models have been quantized to F16  
All models take [1, 112, 112, 3] cropped image of a face as input and produce as single float array as output which represents face embedding

- `insightface-efficientnet-b0`
- `insightface-ghostnet-strides1`
- `insightface-ghostnet-strides2`
- `insightface-mobilenet-emore`
- `insightface-mobilenet-swish`

## Credits

- Original implementation: <https://github.com/deepinsight/insightface>
- Keras port: <https://github.com/leondgarse/Keras_insightface>

## TBD

- Calculate alternate embeddings
- Enable switching method between embeddings
