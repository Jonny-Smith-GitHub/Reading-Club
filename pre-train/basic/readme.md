BERT源码解析参考： https://blog.csdn.net/jiaowoshouzi/article/details/89388794#bertmodel%E7%B1%BB

1.提出

对于Token级任务如问答，双向上下文信息很重要。以前的标准条件语言模型为了使得不看见“未来预测”信息，只能做到单向模型或者浅层双向模型。例如GPT只使用Transformer Decoder部分（“未来信息”被sequence mask遮盖），ELMo只是最后层简单拼接。 BERT使用Transformer Encoder部分并设计了masked language model(MLM)训练目标来解决上述问题，和next sentence prediction(NSP)训练目标来学习句子级信息。

2.模型 2.1 输入/输出表示

输入嵌入由Token嵌入、Segment嵌入和Position嵌入相叠加。其中英文Token嵌入是WordPiece embeddings，大约30,522个词，使得很少遇到OOV词，中文Token嵌入是字级；Segment嵌入用于区分不同句子；Position嵌入补充位置信息。Position嵌入不同于Transformer使用固定正余弦编码，Token嵌入、Segment嵌入和Position嵌入可梯度更新。序列还包括[CLS]，这个Token对应的最后的隐藏状态被用作分类任务的聚合序列表示;[SEP]用于分割两个句子。

2.2 预训练和微调

模型预训练和微调 MLM：想法和CBOW有点类似，预测遮盖单词。不过MLM是预测随机遮盖的15%目标单词，为了减少预训练和微调模型之间因为mask不一致的影响，作者采用策略：80%用[MASK]替换目标单词，10%用随机的单词替换目标单词，10%不改变目标单词。这使得模型不知哪个token被遮盖或者替换掉，迫使每个token学习上下文表征，且偏向于实际观察到的单词，而不仅仅专注于预测遮盖的token。由于仅对15%的tokens进行预测，所以收敛更慢，不过实验表示这样做收益是值得的。 NSP：两个句子，50%的概率是连续的，50%的概率是随机的。由CLS的最终隐藏向量进行预测分类，IsNext或者NotNext，下游微调任务中如图a，b也是这样做。

Experiments and Ablation Studies 11项NLP任务上做了实验。 Ablation Studies：测试了不同预训练目标和不同模型参数的实验效果；测试了使用BERT作为基于特征的方法来使用，连接最后四层隐藏层效果最好，但仍低于基于微调；测试了不同masked策略的效果，masked比率设置在MNLI和NER上的实验结果是否体现了包含SAME或RND对NER这种单句任务和MNLI这种句子级任务的影响？表明同时包含SAME或RND有助于词嵌入具有更好的上下文含义，这对于句子级任务缺稍有负影响。我们如果同论文Transfer Fine-Tuning- A BERT Case中重新训练模型的话，这个比例设置可能对我们EL任务(类似NER任务)会有提升。
4.Transfer Fine-Tuning- A BERT Case 论文证明了句子和短语释义关系有助于句子表征学习。 但因为论文中句子释义和短语释义（三种分类）都是在句子级上做的预训练操作，所以实验结果也显示在Semantic Equivalence和NLI这些句子级任务上都表现良好，但是在Single-Sentence任务上却比BERT-base表现更差。EL任务会更加偏向于NER这种Single-Sentence任务，所以论文的预训练方法的模型可能不太适合于EL任务。

论文中实验证明了在下游微调任务中有限训练集时效果会优于BERT-base，这对于迁移学习特别是标注数据有限的情况非常有利。这可能是论文的短语释义（phrasal paraphrase）任务和特别设计的三分类任务带来的有益影响。
