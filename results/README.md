|文件标识|描述|评分|
|-|-|-|
0914-1|第一次试水，仅根据rotorspeed的参数范围，将拥有范围外rotorspeed的数据视为异常|0.79530736
0915-1|第二次试水，仅根据windspeed的参数范围，将拥有范围外windspeed的数据视为异常|未增加
0915-2|第三次试水，根据windspeed和rotorspeed的参数范围，将拥有范围外rotorspeed和windspeed的数据视为异常|0.80059751
0916-1|在0915-2的基础上，对每个风机的数据分别进行核密度估计，将每个风机概率最低的5%数据视作异常|未增加
0916-2|不根据风机参数清洗，对每个风机的数据分别进行核密度估计，将每个风机概率最低的5%数据视作异常|未增加
0922-1|根据风机参数清洗，对每个风机，使用统一参数进行grid_var清洗，rs_interval_n=10，pw_interval_n=20，k=1，clip_ratio_step=0.1|0.80525639774 
0922-2|根据风机参数清洗，对每个风机，使用统一参数进行grid_var清洗，rs_interval_n=10，pw_interval_n=20，k=0.5，clip_ratio_step=0.1|0.80949535156 
0922-3|根据风机参数清洗，对每个风机，使用统一参数进行grid_var清洗，rs_interval_n=10，pw_interval_n=20，k=0.2，clip_ratio_step=0.1|0.81263058
0923-1|根据风机参数清洗，对每个风机，使用该风机特定的参数进行ew_grid_var清洗，每个风机用于grid_var清洗的参数见ew_grid_var_param.csv|0.81566951100
0925-1|根据风机参数清洗，对每个风机，使用该风机特定的参数进行ed_grid_var清洗，每个风机用于grid_var清洗的参数见ed_grid_var_param.csv（此次未调参）|0.81202082595
0925-2|不进行一次清洗，对每个风机，使用该风机特定的参数进行ew_grid_var清洗，每个风机用于grid_var清洗的参数见ed_grid_var_param.csv（此次未调参）|?
0925-3|不进行一次清洗，对每个风机，使用该风机特定的参数进行ed_grid_var清洗，每个风机用于grid_var清洗的参数见ed_grid_var_param.csv（此次未调参）|?
0927-1|不进行一次清洗，对每个风机，首先将底部异常点去除，随后使用该风机特定的参数进行ed_grid_var清洗，每个风机用于grid_var清洗的参数见ed_grid_var_param.csv|?
0930-1|按照Data-Driven_Correction_Approach_to_Refine_Power_Cu复现，未调参|0.88814290181
0930-2|按照Data-Driven_Correction_Approach_to_Refine_Power_Cu复现，加上去除各维度0以下、去除切入/切出风速外产生功率的点，未调参|0.88739047107
1001-1|按照Data-Driven_Correction_Approach_to_Refine_Power_Cu复现，加上去除各维度0以下的点，未调参|0.88814290
1001-2|在1001-2的基础上，调参，使6号风机在双流形中选取正确的流形|0.88823816
1004-1|结合特定风机处理和顶上直线过采样|0.88823816
1004-1|结合特定风机处理和顶上直线过采样|0.90664124
1004-2|only特定风机处理|0.90664124
1004-3|结合特定风机处理和顶上直线过采样（随机生成）|0.90664124
1005-2|结合特定风机处理（究极大杂烩）和顶上直线过采样（复制生成）|0.90962919048
1005-3|在1005-2的基础上微调了6号风机|?
1006-1|在1005-2的基础上微调了6号风机|?
1006-2|在1006-1的基础上微调了5号风机的参数|0.90974416732
1006-3|在1006-2的基础上调整了9号风机的过采样策略|0.91081493636 
1007-1|在1006-2的基础上调整了2, 6, 8, 9号风机的过采样策略|0.90974416732
1007-2|在1007-1的基础上调整了参数|0.91127697099
1012-1|在1007-2的基础上用规则调整了1号风机|0.91233103544
1012-2|在1012-1的基础上用规则调整了2号风机|0.91233103544
1012-3|在1012-2的基础上用规则调整了3号风机|0.91150943002 
1013-1|在1012-2的基础上用规则调整了4 5 6号风机|0.91302587192
1013-2|在1013-1的基础上用规则调整了7 8号风机|0.91236407597
1013-3|在1013-2的基础上用规则调整了7号风机，没有割去“上方”部分|0.91291695517
1014-1|在1013-1的基础上用规则调整了9 10号风机，没有割去“上方”部分|0.91441174197
1014-2|在1014-1的基础上用规则调整了11 12号风机，没有割去“上方”部分|0.90722786103
1014-3|在1014-2的基础上用规则调整了11风机，没有割去“下方”部分，而是割去“上方”|0.90578229985
memo|3 7号风机有问题！|?
