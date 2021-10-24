# DDH
This is the Tensorflow implementation of paper "Deep Distillation Hashing for Unconstrained Palmprint Recognition". 

## Environment
Tensorflow 1.13.1  
python 3.5

## Training Model and Predicting

Download the pre-trained vgg16.npy <br />
(https://mega.nz/file/YU1FWJrA#O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM)

### Training:  
  
**Trainiing teacher**: train_teacher.py  
**Training student**: train_student.py  
**Training DDH**: train_DDH.py  
  
### Evaluate:  
  
eval.py  
Draw_DET.py  

## The databases used

PolyU multispectical palmprint database <br />
https://www4.comp.polyu.edu.hk/~biometrics/MultispectralPalmprint/MSP.htm <br />
Tongji  palmprint database <br />
http://sse.tongji.edu.cn/linzhang/cr3dpalm/cr3dpalm.htm <br /> 


## Citation

```
@article{DBLP:journals/tim/ShaoZD21,
  author    = {Huikai Shao and
               Dexing Zhong and
               Xuefeng Du},
  title     = {Deep Distillation Hashing for Unconstrained Palmprint Recognition},
  journal   = {{IEEE} Trans. Instrum. Meas.},
  volume    = {70},
  pages     = {1--13},
  year      = {2021},
  url       = {https://doi.org/10.1109/TIM.2021.3053991},
  doi       = {10.1109/TIM.2021.3053991},
  timestamp = {Tue, 02 Mar 2021 11:25:25 +0100},
  biburl    = {https://dblp.org/rec/journals/tim/ShaoZD21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
