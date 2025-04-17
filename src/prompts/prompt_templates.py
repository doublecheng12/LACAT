"""
提示模板模块，包含系统中使用的所有提示模板。
为了方便管理和修改，所有提示都集中在这个文件中。
"""

class PromptTemplates:
    """提示模板集合"""
    
    # 第一次选择提示
    FIRST_PROMPT = """从候选习题中选择一道你认为最具有代表性的习题，从难度，区分度，知识点覆盖度方面考虑，可参考计算机自适应测试的选题原则。
                       只输出试题序号....，不要输出其他任何内容
                       candidate questions: {candidate_questions}
                       Your output MUST follow the following format:
                       ```json 
                       {{"reason": "your reasoning steps","question_index": "the question id what you want to recommend,must be int"}}
                       ```

                       Here are some examples:
                        # 
                        candidate questions: ['什么是集合的∩操作？','已知an = 2n + 1 求Sn','给定两个集合：集合A = {{1, 3, 5, 7, 9}},集合B = {{1, 2, 3, 4, 5}}，求A∩B','已知an = 2n + 1 求Sn']
                        Expected Output:
                        ```json
                        {{"reason": "第2题的习题区分度更高,是很好的初始化选择题目","question_index": 2}}
                        ```
                        candidate questions: ['给定两个集合：集合A = {{1, 3, 5, 7, 9}},集合B = {{1, 2, 3, 4, 5}}，求A∩B','已知SinA = 30,求 e^a 导数','已知an = 2n + 1 求Sn']
                        Expected Output:
                        ```json
                        {{"reason": "第1题的知识点覆盖面更广,包含了三角函数和导数，是很好的初始化选择题目","question_index": 1}}
                        ```
    """
    
    # 生成器提示
    GE_PROMPT = """从候选习题中选择一道最适合学生当前水平的习题，学生在回答完这道习题后能最大化提高学生的能力。请依照计算机自适应测试原则仔细思考。
                       Records: {records}
                       candidate questions: {candidate_questions}
                       Your output MUST follow the following format:
                       ```json 
                       {{"reason": "your reasoning steps","question_index": "the question id what you want to recommend"}}
                       ```

                       Here are some examples:
                        # 
                        Records: 1.已知 集合 A = {{ x | - 2 ≤ x ≤ 5 }} , B = {{ x | m + 1 ≤ x ≤ 2 m - 1 }} . 若 B ⊆ A , 则 实数 m 的 取值 范围 为 {{blank}}\n\tA: m ≤ 3\tB: 2 ≤ m ≤ 3\tC: m ≥ 2\tD: m ≥ 3 错误
                        candidate questions: ['给定两个集合：集合A = {{1, 3, 5, 7, 9}},集合B = {{1, 2, 3, 4, 5}}，求A∩B','什么是集合的∩操作？','已知an = 2n + 1 求Sn']
                        Expected Output:
                        ```json
                        {{"reason": "学生在集合的知识点掌握较差，并且在上一次的问题中回答错误。我需要给学生选择一个难度更低的习题","question_index": 0}}
                        ```
    """
    
    # 认知相关生成器提示
    RGE_PROMPT = """依次考虑知识点覆盖率、难度和区分度，从候选习题中选择最适合当前学生水平的习题。
                     请基于学生的优势和劣势进行仔细思考，选择能够更准确衡量学生能力的习题。
                     如果上一道题学生回答正确，可略微提高难度；回答错误则略微降低难度。
                     
                     Strength: {strength}
                     Weakness: {weakness}
                     candidate questions: {candidate_questions}
                     Last question: {last_question}
                     Last_critic: {answer}
                     
                     Your output MUST follow the following format:
                     ```json 
                     {{"reason": "your reasoning steps","question_index": "the index what you want to recommend,must be int"}}
                     ```
                     
                     请尽可能详细阐述你的选题理由，学生需要一个可信的、完备的选题过程，每次只能选择一道题。
                     
                     Example:
                     Strength: 该学生能够成功计算复杂的积分题目，如∫(x^2 sin(x)) dx，并且能够理解和验证向量运算的基本性质。该学生在处理数列和级数的问题上表现不佳，在求解复杂的递推关系和级数求和方面表现优秀
                     Weakness: 学生在三角函数的图形变换和应用上需要进一步加强。
                     candidate questions: ['给定两个集合：集合A = {{1, 3, 5, 7, 9}},集合B = {{1, 2, 3, 4, 5}}，求A∩B','什么是集合的∩操作？','\(a_n\) 的递推关系为 \(a_{{n+1}} = 2a_n + n^2\)，且已知 \(a_1 = 1\)。']
                     Last question: 考虑数列 \(a_n\)，其递推关系为 \(a_{{n+1}} = 3a_n + 4\)，且已知 \(a_1 = 5\)。求 \(a_n\) 的通项公式，并计算 \(a_{{10}}\) 正确
                     Last critic: 学生在上一次的数列递推习题作答正确，考虑在数列递推这个知识点选择更难的数学题
                     
                     Expected Output:
                     ```json
                     {{"reason": "为了帮助学生加强对数列和级数的理解与计算能力，特别是在处理递推数列和找出其通项公式方面，同时考虑到学生在上一题的作答情况，需要一道难度更大的数列通项习题","question_index": 2}}
                     ```
    """
    
    # IRT模型能力值提示
    THETA_PROMPT = """基于IRT模型获得的学生能力值(theta)和学生的最近作答记录，选择最适合学生当前水平的习题。
                     如果theta小于0表示学生学习较差，theta大于0表示学生学习较好。
                     请从难度、区分度和知识点覆盖度方面考虑，选择能最大化提高学生能力的习题。
                     
                     theta: {theta}
                     candidate questions: {candidate_questions}
                     Last question: {last_question}
                     
                     Your output MUST follow the following format:
                     ```json 
                     {{"reason": "your reasoning steps","question_index": "the index what you want to recommend"}}
                     ```
                     
                     Example:
                     theta = 0.72534
                     candidate questions: ['给定两个集合：集合A = {{1, 3, 5, 7, 9}},集合B = {{1, 2, 3, 4, 5}}，求A∩B','什么是集合的∩操作？','\(a_n\) 的递推关系为 \(a_{{n+1}} = 2a_n + n^2\)，且已知 \(a_1 = 1\)。']
                     Last question: 考虑数列 \(a_n\)，其递推关系为 \(a_{{n+1}} = 3a_n + 4\)，且已知 \(a_1 = 5\)。求 \(a_n\) 的通项公式，并计算 \(a_{{10}}\) 正确
                     
                     Expected Output:
                     ```json
                     {{"reason": "学生的能力值较高，属于高水平学生，同时考虑到学生在上一题的作答情况，需要一道难度更大的数列习题","question_index": 2}}
                     ```
    """
    
    # 长期记忆提示
    LONGTERM_MEMORY_PROMPT = """根据学生的正确和错误答题记录，分析学生的优势和劣势:
                              
                              Correct: {right_answer_records}
                              Incorrect: {wrong_answer_records}
                              
                              请基于正确答案记录总结学生的优势，基于错误答案记录总结学生的劣势。
                              你的总结应当简洁明了，不要添加其他内容，也不要简单重复已有信息。
                              
                              请以JSON格式输出:
                              ```json
                              {{"strength": "基于正确答案记录总结的优势",
                               "weakness": "基于错误答案记录总结的劣势"}}
                              ```
                              
                              请尽可能详细，学生需要一个完整、可信的画像描述。
                              
                              示例:
                              Input:
                              ```
                              Correct: ['计算积分 ∫(x^2 sin(x)) dx', '验证向量点乘的交换律 a·b = b·a']
                              Incorrect: ['求导函数 f(x) = e^(x^2) 的导数', '使用余弦定理解三角形 ABC，给定边长 a=3, b=4, C=90°', '计算圆柱体的体积，已知底面半径 r=3, 高 h=5']
                              ```
                              
                              Output:
                              ```json
                              {{
                                "strength": "对积分技巧和三角函数在积分中的应用有一定的理解。",
                                "weakness": "该学生在求解复合函数的导数，应用几何定理解题存在着一定劣势。"
                              }}
                              ```
    """
    
    # 短期记忆提示
    SHORT_MEMORY_PROMPT = """基于学生的个人简介和最近的答题记录，理解学生的需求并推测他们在最近的问题背景下需要回答的问题。
                            
                            Profile: {profile}
                            Last question record: {last_question}
                            
                            基于计算机自适应测试的原则，简洁地给出你的答案。
                            以JSON格式输出:
                            
                            ```json
                            {{"thought": "你的思考过程"}}
                            ```
    """
    
    # 评论家提示
    CRITIC_PROMPT = """作为检查员，检查学生习题推荐中的基本错误。以下是一些典型错误类型和相应示例：
                      
                      错误类型示例：
                      1. 推荐的习题难度过大，超过学生的能力  
                         student_profile: 学生在三角函数的知识点的掌握程度较差，在数列知识点方面掌握良好
                         last_recommended_exercise: 已知函数 $f(x) = 3\sin(2x - \frac{{\pi}}{{3}}) + 4\cos(x + \frac{{\pi}}{{6}}) - 2\sin^2x$，求函数 $f(x)$ 的单调递增区间。
                         explanation: 最近一次的习题推荐在三角函数知识点选择的难度过大，应该降低该知识点的习题难度
                         
                      2. 推荐的习题难度过低，低于学生的真实能力
                         student_profile: 学生在三角函数的知识点的掌握程度较好，在数列知识点方面掌握一般
                         last_recommended_exercise: 已知函数 $f(x) = sin2x，求函数 $f(x)$ 的单调递增区间。
                         explanation: 最近一次的习题推荐在三角函数知识点选择的难度过低，学生在三角函数方面掌握已经较好，应该选择更高难度的习题
                         
                      3. 推荐的习题知识点饱和
                         student_profile: 学生在三角函数的知识点的掌握程度较好，在数列知识点方面掌握很好
                         last_recommended_exercise: 已知函数 $f(x) = 3\sin(2x - \frac{{\pi}}{{3}}) + 4\cos(x + \frac{{\pi}}{{6}}) - 2\sin^2x$，求函数 $f(x)$ 的单调递增区间。
                         explanation: 推荐的习题知识点饱和，应该推荐其他知识点的习题。
                      
                      请检查以下学生推荐是否存在上述错误：
                      student_profile: {profile}
                      last_recommended_exercise: {exercise}
                      
                      请检查推荐是否存在上述错误并给出你的判断和反馈。你的反馈可以类似于上面的解释。
                      以JSON格式输出：
                      
                      ```json
                      {{"judgement": "错误类型", "feedback": "你希望给出的简短提醒"}}
                      ```
    """
    
    # 预测提示
    PREDICT_PROMPT = """作为认知诊断模型，基于学生个人简介预测学生是否能正确回答问题，1表示正确，0表示错误。
                       
                       示例：
                       1. 
                       student_profile: 学生在三角函数的知识点的掌握程度较差，在数列知识点方面掌握良好
                       question: 已知函数 $f(x) = 3\sin(2x - \frac{{\pi}}{{3}}) + 4\cos(x + \frac{{\pi}}{{6}}) - 2\sin^2x$，求函数 $f(x)$ 的单调递增区间。
                       {{"answer" : 0}}
                       
                       2.
                       student_profile: 学生在三角函数的知识点的掌握程度较好，在数列知识点方面掌握良好
                       last_recommended_exercise: 已知函数 $f(x) = sin2x，求函数 $f(x)$ 的单调递增区间。
                       {{"answer" : 1}}
                       
                       以下是需要预测的学生回答：
                       
                       student_profile: {profile}
                       question: {question}
                       
                       以JSON格式输出：
                       
                       ```json
                       {{"answer": 1 或 0}}
                       ```
    """