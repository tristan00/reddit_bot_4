# reddit_bot_4

reddit bot 1, naive method similar to localized regression, some success on the simpler subreddits
reddit bot 3, dnn to clasifiy comments, would attempt to reuse comments, rank them  by classification post the top one, no success
reddit bot 4, lstm to generate comments, some regression algorithm to predict success


Experimented with word vectors into lstm network, it was not learning much. will work on lstm design.

Experimenting with letter lstms and dnns, mixed success. Will attempt huge dnn and better designed lstm.

For score prediction:
Using only metadata, had best accuracy with extra tree random forests and grandient boosting regressors. extra tree method took 1/3 the time and had lower variance. Will do more testing.

Considering introducing word vector features or topic models as features.
