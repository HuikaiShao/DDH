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
@article{DBLP:journals/corr/abs-2004-03303,
  author    = {Huikai Shao and
               Dexing Zhong and
               Xuefeng Du},
  title     = {Towards Efficient Unconstrained Palmprint Recognition via Deep Distillation
               Hashing},
  journal   = {CoRR},
  volume    = {abs/2004.03303},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.03303},
  archivePrefix = {arXiv},
  eprint    = {2004.03303},
  timestamp = {Wed, 08 Apr 2020 17:08:25 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-03303.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
