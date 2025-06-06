import asyncio
from abc import ABC, abstractmethod
from code.config import Config
from code.llm import LLMProvider
import json
from pydantic import BaseModel, Field, RootModel
from typing import Dict, List, Union, Optional
import math
import random

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, state, parent=None):
        self.state = state  # 当前场景状态
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.score = 0  # 累计评分
        self.uct = 0  # UCT值

    def expand(self, possible_states):
        """扩展节点"""
        for state in possible_states:
            child = MCTSNode(state, self)
            self.children.append(child)

    def update(self, score):
        """更新节点统计信息"""
        self.visits += 1
        self.score += score
        if self.parent:
            self.uct = (self.score / self.visits) + math.sqrt(2 * math.log(self.parent.visits) / self.visits)

    def select_best_child(self, exploration_weight=1.41):
        """选择最佳子节点"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.uct)

class MCTSGenerator:
    """蒙特卡洛树搜索生成器"""
    def __init__(self, scriptwriter_agent):
        self.agent = scriptwriter_agent
        self.max_iterations = 5  # 迭代次数
        self.exploration_weight = 1.41
        self.max_depth = 3  # 限制搜索深度
        self.cache = {}  # 添加缓存

    async def generate_scene(self, script, gamelog):
        """使用MCTS生成场景"""
        root = MCTSNode(None)
        iteration_count = 0
        model_calls = 0
        
        while iteration_count < self.max_iterations:
            # 选择阶段
            node = self._select(root)
            depth = 0
            current = node
            while current.parent:
                depth += 1
                current = current.parent
            
            # 如果达到最大深度，跳过扩展
            if depth >= self.max_depth:
                continue
                
            # 扩展阶段
            if node.visits > 0:
                possible_states = await self._get_possible_states(node.state, script, gamelog)
                model_calls += len(possible_states)  # 记录生成调用
                node.expand(possible_states)
                if node.children:
                    node = random.choice(node.children)
            
            # 模拟阶段
            cache_key = str(node.state)
            if cache_key in self.cache:
                score = self.cache[cache_key]
            else:
                score = await self._simulate(node.state, script, gamelog)
                model_calls += 1  # 记录评估调用
                self.cache[cache_key] = score
            
            # 反向传播阶段
            while node:
                node.update(score)
                node = node.parent
                
            iteration_count += 1
            print(f'迭代 {iteration_count}/{self.max_iterations}, 模型调用次数: {model_calls}')
        
        # 选择最佳场景
        best_node = root.select_best_child()
        return best_node.state if best_node else None

    def _select(self, node):
        """选择阶段"""
        while node.children:
            if not all(child.visits > 0 for child in node.children):
                return random.choice([c for c in node.children if c.visits == 0])
            node = node.select_best_child(self.exploration_weight)
        return node

    async def _get_possible_states(self, current_state, script, gamelog):
        """获取可能的场景状态"""
        # 使用LLM生成多个可能的场景变体
        states = []
        for _ in range(2):  # 减少变体数量
            state = await self.agent._generate_scene(script, gamelog)
            states.append(state)
        return states

    async def _simulate(self, state, script, gamelog):
        """模拟阶段"""
        if not state:
            return 0
        # 评估场景质量
        score = await self.agent._evaluate_scene(state, script, gamelog)
        return score




class SceneContent(BaseModel):
    """场景内容模型"""
    场景: str
    人物: str
    情节链: List[str]
    流: Dict[str, List[Union[str, Dict[str, str]]]]
    交互: Dict[str, List[str]]
    触发: Dict[str, Dict[str, Union[str, List[str]]]]

class SceneOutput(RootModel):
    """场景输出模型"""
    root: Dict[str, Union[SceneContent, Dict[str, str]]]

class EvaluationOutput(BaseModel):
    """评估输出模型"""
    score: int = Field(ge=0, le=5)
    reason: str

class BaseScriptwriterAgent(ABC):
    def __init__(
        self,
        llm_model=Config.DRAMA_AGENT_MODEL_NAME,
        llm_provider=Config.DRAMA_AGENT_MODEL_PROVIDER,
    ):
        self._llm_model = llm_model
        self._llm_provider = LLMProvider(provider=llm_provider)

    async def gen_new_full_script(self) -> dict:
        """
        生成全部场景剧本（暂时待定）
        """
        pass

    @abstractmethod
    async def gen_new_scene_script(self, script: dict, gamelog: dict) -> dict:
        """
        生成一幕新场景剧本
        """
        pass



class ScriptwriterAgent(BaseScriptwriterAgent):
    def __init__(
        self,
        llm_model=Config.DRAMA_AGENT_MODEL_NAME,
        llm_provider=Config.DRAMA_AGENT_MODEL_PROVIDER,
    ):
        super().__init__(llm_model, llm_provider)
        self.mcts_generator = MCTSGenerator(self)

    async def gen_new_scene_script(self, script=None, gamelog=None):
        """
        使用MCTS生成新的场景剧本
        """
        # 使用MCTS生成场景
        new_scene = await self.mcts_generator.generate_scene(script, gamelog)
        
        if new_scene:
            print('MCTS生成成功')
            return new_scene
        else:
            # 如果MCTS生成失败，回退到原始生成方法
            print('MCTS生成失败，回退到原始生成方法')
            return await self._generate_scene(script, gamelog)

    async def _generate_scene(self, script, gamelog):
        """
        使用LLM生成新的场景剧本
        """
        # 准备提示词
        prompt = SCENE_GEN_PROMPT_TEMP.format(
            script=json.dumps(script, ensure_ascii=False),
            plot_history=json.dumps(gamelog.get("plot_history", []), ensure_ascii=False),
            clue_history=json.dumps(gamelog.get("clue_history", []), ensure_ascii=False),
            hint_history=json.dumps(gamelog.get("hint_history", []), ensure_ascii=False),
            interaction_history=json.dumps(gamelog.get("interaction_history", []), ensure_ascii=False)
        )
        
        # 调用LLM生成场景
        response = await self._llm_provider.infer(
            model=self._llm_model,
            prompt=prompt,
            response_model=SceneOutput
        )
        print('新场景的结果：')
        print(json.dumps(response, ensure_ascii=False, indent=2))
        
        try:
            # 直接返回字典
            return response
        except Exception as e:
            print(f"场景生成失败：{str(e)}")
            return await self._dummy_gen_new_scene_script(script, gamelog)

    async def _evaluate_scene(self, new_scene, script, gamelog):
        """
        评估新生成的场景
        """
        # 准备评估提示词
        prompt = SCENE_EVAL_PROMPT_TEMP.format(
            script=json.dumps(script, ensure_ascii=False),
            plot_history=json.dumps(gamelog.get("plot_history", []), ensure_ascii=False),
            clue_history=json.dumps(gamelog.get("clue_history", []), ensure_ascii=False),
            hint_history=json.dumps(gamelog.get("hint_history", []), ensure_ascii=False),
            interaction_history=json.dumps(gamelog.get("interaction_history", []), ensure_ascii=False),
            new_scene=json.dumps(new_scene, ensure_ascii=False)
        )
        
        # 调用LLM进行评估
        response = await self._llm_provider.infer(
            model=self._llm_model,
            prompt=prompt,
            response_model=EvaluationOutput
        )
        print('评分的结果：')
        print(json.dumps(response, ensure_ascii=False, indent=2))
        
        try:
            # 直接获取评分
            return response.get("score", 0)
        except Exception as e:
            print(f"场景评估失败：{str(e)}")
            return 0
        

    async def _dummy_gen_new_scene_script(self, script=None, gamelog=None):
 
        await asyncio.sleep(1)
        return {
            "场景老王烧饼铺": {
                "场景": "地点：老王烧饼铺\\n时间：上午十点\\n你来到隔壁老王的烧饼铺，蒸笼冒着热气却未见武大郎的摊位。",
                "人物": "老王。隔壁老王四十余岁，满脸横肉，手臂有烫伤疤痕。因摊位纠纷与武大郎积怨已久，近日正在争夺早市黄金摊位。",
                "情节链": [
                    "潘金莲试探老王与武大郎的矛盾",
                    "老王察觉潘金莲异常神色",
                    "潘金莲试图用砒霜栽赃老王",
                    "老王反咬潘金莲通奸之事",
                    "烧饼铺伙计目击争执",
                ],
                "流": {
                    "潘金莲试探老王与武大郎的矛盾": [
                        "老王（擦着擀面杖）：武大家的？稀客啊，你家那矮子今天怎舍得让娇妻抛头露面？",
                        "潘金莲：王大哥说笑了，奴家来问问前日您说要买我家祖传和面方子的事...",
                        {"关键提示": "老王右手虎口有新鲜抓痕"},
                    ],
                    "老王察觉潘金莲异常神色": [
                        "老王（突然逼近）：你袖口沾的可是石灰？今早西巷棺材铺刚运走三袋。",
                        "潘金莲（后退半步）：王大哥真会说笑，这是...揉面沾的面粉。",
                        {"收集关键线索": "老王注意到潘金莲袖口异常"},
                    ],
                    "潘金莲试图用砒霜栽赃老王": [
                        "潘金莲（掏出纸包）：其实奴家是想问问，王大哥面案下藏的砒霜可要分些与奴家？",
                        "老王（拍案而起）：好个毒妇！昨日武大说要去县衙告我强占摊位，今早就...",
                        {"关键提示": "蒸笼后闪过烧饼铺伙计的身影"},
                    ],
                    "老王反咬潘金莲通奸之事": [
                        "老王（阴笑）：上月廿八未时，西门大官人的马车在你家后巷停了一炷香。",
                        "潘金莲（脸色煞白）：你...你血口喷人！",
                        {"收集关键线索": "老王掌握潘金莲与西门庆私会证据"},
                    ],
                    "烧饼铺伙计目击争执": [
                        "伙计（突然插话）：掌柜的，武家娘子方才在面缸旁鬼鬼祟祟...",
                        "潘金莲（猛然转身）：休得胡言！",
                        {"关键提示": "面缸边缘有白色粉末洒落"},
                    ],
                },
                "交互": {
                    "对话": [
                        "潘金莲提及摊位纠纷$语义1 (武大郎这两天和隔壁老王产生过巨大矛盾)",
                        "老王暗示知晓武大郎死亡真相$语义2 (老王注意到潘金莲袖口异常)",
                        "烧饼铺伙计指认可疑行为$语义3 (面缸边缘有白色粉末洒落)",
                    ],
                    "动作选择": [
                        "摔碎毒药瓶诬陷老王$1 (潘金莲试图用砒霜栽赃老王)",
                        "谎称武大郎去县衙告状$2 (潘金莲提及摊位纠纷$语义1)",
                        "用西门庆势力威胁老王$3 (老王掌握潘金莲与西门庆私会证据)",
                        "借口取面粉查看面缸$4 (烧饼铺伙计指认可疑行为$语义3)",
                    ],
                },
                "触发": {
                    "摔碎毒药瓶诬陷老王$1": {
                        "叙事": "你故意打翻砒霜纸包，白色粉末飘向正在发酵的面团。",
                        "收集关键线索": "老王的面团沾染不明粉末",
                        "跳转": "结局18",
                    },
                    "谎称武大郎去县衙告状$2": {
                        "叙事": "你声称武大郎正在县衙办理摊位过户文书，老王抄起砍骨刀冲向衙门。",
                        "收集关键线索": "老王持凶器前往县衙",
                        "跳转": "场景大街",
                    },
                    "用西门庆势力威胁老王$3": {
                        "叙事": "你暗示西门庆会处理多嘴之人，老王狂笑着掀开藏着账本的面缸。",
                        "收集关键线索": "发现老王走私面粉的暗账",
                        "跳转": "场景西门庆家",
                    },
                    "借口取面粉查看面缸$4": {
                        "叙事": "你假装查看面粉质量，将武大郎的鞋底碎布塞入缸底。",
                        "收集关键线索": "老王面缸发现武大郎衣物残片",
                        "跳转": "场景衙门",
                    },
                },
            },
            "结局18": {
                "流": "老王的面团被验出砒霜，但在衙役搜查时发现你袖中相同的药包纸，最终两人以互投毒罪收监。"
            },
        }


SCENE_GEN_PROMPT_TEMP = """
你是一个专业的剧本作家，需要根据给定的游戏历史和当前剧本生成新的场景。请严格按照以下格式生成剧本：

当前剧本：
{script}

玩家游戏历史：
- 历史剧情：{plot_history}
- 历史线索：{clue_history}
- 历史提示：{hint_history}
- 历史交互：{interaction_history}

请根据剧情发展生成新的场景剧本，要求：
1. 场景名称格式：
   - 普通场景：使用"场景XXX"格式
   - 结局场景：使用"结局XX"格式（仅在剧情需要结局时生成）

2. 普通场景必须包含以下字段：
   - 场景：描述场景的时间、地点和基本环境
   - 人物：描述场景中的人物及其特征
   - 情节链：该场景可能发生的情节列表
   - 流：每个情节的具体对话和事件
   - 交互：玩家可选的对话和动作选项
   - 触发：不同交互可能触发的剧情走向

3. 结局场景格式：
   - 场景名称必须以"结局"开头
   - 只需包含"流"字段，描述完整的结局内容
   - 根据剧情发展决定是否需要生成结局

4. 新生成的场景必须：
   - 与历史剧情保持连贯
   - 合理利用已收集的线索
   - 符合人物性格特征
   - 提供有意义的剧情发展
   - 保持悬疑感和戏剧性

请按照以下JSON格式输出：
{{
    "场景名称": {{
        "场景": "场景描述",
        "人物": "人物描述",
        "情节链": ["情节1", "情节2", ...],
        "流": {{
            "情节1": [
                "对话1",
                "对话2",
                {{"关键提示": "提示内容"}},
                {{"收集关键线索": "线索内容"}}
            ]
        }},
        "交互": {{
            "对话": [
                "对话选项1$语义1 (触发条件1)",
                "对话选项2$语义2 (触发条件2)"
            ],
            "动作选择": [
                "动作1$1 (触发条件1)",
                "动作2$2 (触发条件2)"
            ]
        }},
        "触发": {{
            "动作1$1": {{
                "叙事": "触发后的叙事内容",
                "收集关键线索": "新的线索",
                "跳转": "下一个场景"
            }}
        }}
    }}
}}

如果剧情需要结局，则额外添加结局场景：
{{
    "结局XX": {{
        "流": "结局的完整描述"
    }}
}}

示例：
1. 普通场景：
{{
    "场景老王烧饼铺": {{
        "场景": "地点：老王烧饼铺\\n时间：上午十点\\n你来到隔壁老王的烧饼铺，蒸笼冒着热气却未见武大郎的摊位。",
        "人物": "老王。隔壁老王四十余岁，满脸横肉，手臂有烫伤疤痕。因摊位纠纷与武大郎积怨已久，近日正在争夺早市黄金摊位。",
        "情节链": ["潘金莲试探老王与武大郎的矛盾", "老王察觉潘金莲异常神色"],
        "流": {{
            "潘金莲试探老王与武大郎的矛盾": [
                "老王（擦着擀面杖）：武大家的？稀客啊，你家那矮子今天怎舍得让娇妻抛头露面？",
                "潘金莲：王大哥说笑了，奴家来问问前日您说要买我家祖传和面方子的事...",
                {{"关键提示": "老王右手虎口有新鲜抓痕"}}
            ]
        }},
        "交互": {{
            "对话": [
                "潘金莲提及摊位纠纷$语义1 (武大郎这两天和隔壁老王产生过巨大矛盾)",
                "老王暗示知晓武大郎死亡真相$语义2 (老王注意到潘金莲袖口异常)"
            ],
            "动作选择": [
                "摔碎毒药瓶诬陷老王$1 (潘金莲试图用砒霜栽赃老王)",
                "谎称武大郎去县衙告状$2 (潘金莲提及摊位纠纷$语义1)"
            ]
        }},
        "触发": {{
            "摔碎毒药瓶诬陷老王$1": {{
                "叙事": "你故意打翻砒霜纸包，白色粉末飘向正在发酵的面团。",
                "收集关键线索": "老王的面团沾染不明粉末",
                "跳转": "结局18"
            }}
        }}
    }}
}}

2. 结局场景（仅在需要时生成）：
{{
    "结局18": {{
        "流": "老王的面团被验出砒霜，但在衙役搜查时发现你袖中相同的药包纸，最终两人以互投毒罪收监。"
    }}
}}
"""

SCENE_EVAL_PROMPT_TEMP = """
你是一个专业的剧本评估专家，需要评估新生成的场景与原有剧情及玩家历史的关联性。请根据以下标准进行评分：

当前剧本：
{script}

玩家游戏历史：
- 历史剧情：{plot_history}
- 历史线索：{clue_history}
- 历史提示：{hint_history}
- 历史交互：{interaction_history}

新生成的场景：
{new_scene}

评分标准（0-5分）：
0分：完全无关，与原有剧情和玩家历史毫无联系
1分：勉强相关，只有少量元素与原有剧情或玩家历史有关
2分：部分相关，能利用部分历史线索和剧情发展
3分：基本相关，较好地利用了历史线索，剧情发展合理
4分：高度相关，充分利用历史线索，剧情发展自然流畅
5分：完美相关，完美整合所有历史元素，剧情发展既出人意料又合情合理

具体评估维度：
1. 剧情连贯性：
   - 新场景是否自然承接原有剧情
   - 人物行为是否符合之前的发展
   - 情节转折是否合理

2. 线索利用：
   - 是否合理利用已收集的线索
   - 是否产生新的有价值的线索
   - 线索之间的关联是否合理

3. 人物表现：
   - 人物性格是否保持一致
   - 对话风格是否统一
   - 行为动机是否合理

4. 剧情发展：
   - 是否推动故事向前发展
   - 是否保持悬疑感和戏剧性
   - 是否提供有意义的剧情选择

请按照以下JSON格式输出评分：
{{
    "score": 分数,
    "reason": "评分理由，包括对各个维度的具体分析"
}}

示例输出：
{{
    "score": 4,
    "reason": "新场景很好地利用了玩家收集的线索，特别是关于老王与武大郎的矛盾。剧情发展自然，通过烧饼铺的场景展现了新的矛盾点。人物性格保持一致，对话风格统一。唯一不足是结局略显仓促，可以进一步展开。"
}}
"""
