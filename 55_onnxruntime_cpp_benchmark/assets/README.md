# Day55 benchmark assets

These six images are a small, reviewable input set for measuring image decoding, preprocessing, and ONNX Runtime CPU inference. They are not accuracy, class-coverage, or external-generalization evidence for the forest-species model.

| Tracked file | Original local source | Size (width x height) | Format | Selection reason | SHA256 |
| --- | --- | ---: | --- | --- | --- |
| `banana_bird_256x256.png` | `D:\DL_code\data\banana-detection\bananas_train\images\0.png` | 256 x 256 | PNG | Square medium input with natural background detail | `924E832F414652E778176765450E7907915FD6508B18F2A68088EA58C13E1E8E` |
| `hotdog_wrap_398x296.png` | `D:\DL_code\data\hotdog\test\hotdog\1003.png` | 398 x 296 | PNG | Landscape food image with moderate texture | `AF497317679217E257B495780C38807AA549D729655929F228CCFE211881416B` |
| `fruit_strip_140x60.png` | `D:\DL_code\data\hotdog\test\not-hotdog\1001.png` | 140 x 60 | PNG | Very wide, small input that exercises resize-short-side behavior | `26D46F572CE65981839ADFD538E56F48629F12C823734833B2A35208F634BC5B` |
| `dog_grooming_400x301.jpg` | `D:\DL_code\data\kaggle_dog_tiny\test\0dc570ec7086bab004a7e357164c04b8.jpg` | 400 x 301 | JPEG | JPEG decode path and a cluttered natural scene | `A1CBAC0D6257A33C99D921E71E3D54B4132BBA402800C24AA42018AACE031CBB` |
| `cifar_airplane_32x32.png` | `D:\DL_code\data\kaggle_cifar10_tiny\test\126979.png` | 32 x 32 | PNG | Extremely small input that requires substantial upscaling | `7E812D9CB94D3F16069E734E57A39990CAF7A20F69AD699B67CF7403529BD08D` |
| `hotdog_pickle_614x419.png` | `D:\DL_code\data\hotdog\test\hotdog\1013.png` | 614 x 419 | PNG | Largest selected image and highest tracked byte size | `20168B9E708F2EFFA262D3393141855048A571BA5A4CC39AB96EFD9DA31EDBAE` |

The originals were already present in the learner's D2L course-data pool. Only these six samples are copied; the full datasets are not duplicated. Their semantic classes are intentionally irrelevant to the Day55 performance claim.
