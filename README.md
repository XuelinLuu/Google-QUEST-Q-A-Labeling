#### Google QUEST Q&A Labeling

- 项目是对kaggle题目Google QUEST Q&A Labeling的练习
    - https://www.kaggle.com/c/google-quest-challenge
- 通过使用BERT模型来进行训练

---
- 项目介绍
    - bert-base-tiny-uncased
        - 本人使用的是tiny bert，参数比较少，训练比较快，但在上传的项目中已经全部修改为bert-base-uncased，如果想使用tiny bert，可以修改bert模型的相应位置
    - input
        - 保存项目中输入数据
    - models
        - 保存每一轮项目的输出结果
    - src
        - 项目的各种文件
- 项目运行，直接运行src/train.py
