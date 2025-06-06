###############################################################
# 请不要修改以下任何代码，否则会影响自动化测评（如有问题请与我们联系）
###############################################################

import asyncio
import random
from code.config import Config
from code.llm import LLMProvider
from code.scriptwriter import ScriptwriterAgent
from typing import List

import yaml
from pydantic import BaseModel, Field


class RelationOutput(BaseModel):
    relation: int = Field(
        description="用一个数字代表两个人的合作关系，范围为0到100之间"
    )
    reason: str = Field(description="理由，为什么对合作关系数值进行修改")


class NLUOutput(BaseModel):
    intention: str = Field(description="意图，必须是给定的意图候选项中的一项")
    reason: str = Field(description="理由，为什么选择这个意图选项")


class ReactOutput(BaseModel):
    dialogue: str = Field(
        description="回应潘金莲的对话和行为，格式为 '扮演的角色（包含动作、神态、心理活动等行为）：对话内容', 例如：'潘金莲（起身整理了下衣服，面带笑容的迎面走来）：你好！'",
        min_lenth=1,
        max_length=100,
    )


class NLUandReactOutput(BaseModel):
    intention: str = Field(description="意图，必须是给定的意图候选项中的一项")
    dialogue: str = Field(
        description="回应潘金莲的对话和行为，格式为 '扮演的角色（包含动作、神态、心理活动等行为）：对话内容', 例如：'潘金莲（起身整理了下衣服，面带笑容的迎面走来）：你好！'",
        min_lenth=1,
        max_length=100,
    )


class CueOutput(BaseModel):
    narration: str = Field(description="剧情旁白，不要出现任何人物对话", max_length=150)
    dialogue: List[str] = Field(
        description="人物对话列表，每个元素为一个人物的对话，例如：['潘金莲（起身整理了下衣服，面带笑容的迎面走来）：你好！', '王婆（笑嘻嘻迎上来）：你好！']",
        max_length=2,
    )


class JudgeOutput(BaseModel):
    plotchain: str = Field(description="剧本情节链中的某一节点")
    is_comp: bool = Field(description="是否完成该剧情链节点对应流中的大部分剧情")


class PrereplyOutput(BaseModel):
    intention: str = Field(description="潘金莲的意图")
    reply: str = Field(description="潘金莲口吻的话术", max_length=50)


class PrerepliesOutput(BaseModel):
    prereplies: List[PrereplyOutput] = Field(
        description="所有可能的意图以及话术", min_length=2, max_length=3
    )


RELATION_PROMPT_TEMP = """
对话与故事：
{plot}

以上是'{roles}'这两个人的对话和故事发展，请你总结目前两个人的合作关系(初始关系为50)，并给出相应的Relation数值(范围0-100)。Relation数值的标准如下:
80: 两人关系好或狼狈为奸，最终确定了在某件事上的合作计划
60: 两人愿意帮助彼此，在某件事上互相合作
50: 两人关系一般，没有利害关系
30: 两人互相争吵、互相威胁，完全不合作
10: 两个人关系为生死仇敌，互相有谋害对方的想法
注意辱骂等词汇会大大降低Relation数值。
请反思你的判断，如果不正确及时纠正。
"""

NLU_PROMPT_TEMP = """
剧情：
{plot}

你是一个互动戏剧导演，根据给定的剧情，判断当前【{role}】角色的意图是否能够匹配到以下意图候选项中，如果匹配，则输出对应的意图选项，并说明理由，如果都不匹配，则输出“无匹配意图”：
{intentions}

请反思你的选择，如果不正确及时纠正。
"""

REACT_PROMPT_TEMP = """
剧本：
{script}

剧情：
{plot}

玩家输入：
潘金莲：{input}

根据以上给定的剧本和剧情，你扮演剧本内的【{role}】与玩家扮演的潘金莲对话，{role}当前心情是{emotion}，你需要根据玩家输入以【{role}】口吻做出适当回复(50字以内)，引导玩家按着情节链和流中的剧情走向进行游戏：
1.充分理解并熟记当前剧本，作为后续对话内容的依据，并根据人物设定和场景主线，扮演场景内的人物与玩家对话，必须回复玩家。
2.根据剧情和玩家输入，按照情节链中【{node}】节点相应流的剧情走向回应玩家，在当前场景中做出正确的剧情引导。
3.剧本中的流是一个用于参考的剧情发展，当玩家输入与流中不同时，你应该根据玩家输入进行合适、上下文连贯的回复，而不是照抄流中的剧情。
4.请反思你的回复，如果不正确及时纠正。
"""

NLU_AND_REACT_PROMPT_TEMP = """
剧本：
{script}

剧情：
{plot}

意图：
{intentions}

根据以上给定的剧本和剧情，你扮演一个互动戏剧导演，完成以下两个任务：
1.判断当前【{role}】角色的意图是否能够匹配到以上意图候选项中，如果匹配，则输出对应的意图选项，并说明理由，如果都不匹配，则输出“无匹配意图”
2.扮演剧本内的【{npc}】与玩家扮演的潘金莲对话，{npc}当前心情是{emotion}，你需要根据玩家输入以【{npc}】口吻做出适当回复，引导玩家按照情节链中【{node}】节点相应流的剧情走向进行游戏，剧本中的流是一个用于参考的剧情发展，当玩家输入与流中不同时，你应该根据玩家输入进行合适、上下文连贯的回复，而不是照抄流中的剧情。

请反思你的选择，如果不正确及时纠正。
"""

ClOSING_REACT_PROMPT_TEMP = """
剧本：
{script}

剧情：
{plot}

玩家输入：
潘金莲：{input}

根据以上给定的剧本和剧情，你扮演剧本内的【{role}】与玩家扮演的潘金莲对话，{role}当前心情是{emotion}，你需要根据玩家输入以【{role}】口吻做出适当回复(50字以内)，并委婉表明自己有事不想继续聊了，建议玩家去往其他地方或者找其他人聊聊（不要出现具体人名）。请反思你的回复，如果不正确及时纠正。
"""

CUE_PROMPT_TEMP = """
剧本：
{script}

剧情：
{plot}

根据以上给定的剧本和剧情，你扮演一个互动戏剧导演，引导玩家按着情节链和流中的剧情走向进行互动：
1.充分理解并熟记当前剧本，作为后续思考的依据，根据剧情续写一小段引导玩家触发情节链中下一节点【{next_plot}】的旁白剧情，并用剧本内的人物口吻进行适当对话提示玩家后续剧情。
2.叙事要承接剧情，过渡自然，禁止输出与剧情中已经存在或者重复的内容，但要体现主要人物和主要剧情。
3.剧本中的流是一个用于参考的剧情发展，当玩家输入与流中不同时，你应该根据玩家输入进行合适、上下文连贯的回复，而不是照抄流中的剧情。
4.请反思你的回复，如果不正确及时纠正。
"""

JUDGE_PROMPT_TEMP = """
剧本：
{script}

剧情：
{plot}

根据以上给定的剧本和剧情，你扮演一个互动戏剧导演，判断当前剧情是否已经完成情节链中【{node}】的大部分剧情。请说明判断理由，并反思你的判断，如果不正确及时纠正。
"""

PREREPLY_PROMPT_TEMP = """
剧本：
{script}

剧情：
{plot}

提示：
{hint}

候选意图：
{intentions}

根据以上给定的剧本和剧情，你扮演一个互动戏剧导演，请从候选意图中选择潘金莲角色在给定的提示下最可能选择的若干项意图以及用潘金莲的口吻表达该意图相应的话术。注意生成的话术(30字以内)要承接当前剧情上下文，内容必须要紧扣对应的候选意图，但是不要完全重复剧情中已有的对话内容。请反思你的判断，如果不正确及时纠正。
"""


class DramaAgent:
    def __init__(
        self,
        script_path="",
        llm_model=Config.DRAMA_AGENT_MODEL_NAME,
        llm_provider=Config.DRAMA_AGENT_MODEL_PROVIDER,
        open_dynamic_script=False,
        show_prompt=False,
    ):
        self._init_llm_pool(llm_model)
        self._llm_provider = LLMProvider(provider=llm_provider)
        self._open_dynamic_script = open_dynamic_script
        self._show_prompt = show_prompt
        self._scriptwriter = ScriptwriterAgent()

        with open(script_path, "r", encoding="utf-8") as f:
            # 这里是整个场景文件，保存为原始的剧本
            self.script_ = yaml.load(f, Loader=yaml.FullLoader)

        self.curr_relation = 50

        self._init_script(script=self.script_.copy())  # 拷贝一个剧本的副本

        all_clues, all_actions, all_intentions = [], [], []

        for scene in self.scenes:
            all_actions += list(self.action_conditions[scene].keys())
            all_intentions += list(self.intention_conditions[scene].keys())

            for node in self.script[scene]["流"]:
                for flow in self.script[scene]["流"][node]:
                    if isinstance(flow, dict) and "收集关键线索" in flow:
                        all_clues += [flow["收集关键线索"]]

            for trigger in self.script[scene]["触发"]:
                if "收集关键线索" in self.script[scene]["触发"][trigger]:
                    all_clues += [self.script[scene]["触发"][trigger]["收集关键线索"]]

        self.gameinfo = {
            "all_clues": all_clues,
            "all_actions": all_actions,
            "all_intentions": all_intentions,
            "all_conditions": list(set(all_clues + all_actions + all_intentions)),
        }
        self.plot_history = {}
        self.clue_history = []
        self.hint_history = []
        self.scene_history = []
        self.plotnode_history = []
        self.interaction_history = []
        self.log = {}
        self.checkpoints = {}
        self._reset_scen(scene="序章")

    def _init_script(self, script=None):
        self.script = script  # 当前的剧本
        self.scenes = [x for x in self.script.keys() if not x.startswith("结局")]

        self.ending = {}   # 存放所有的结局，k表示结局i
        for k in self.script:
            if k.startswith("结局"):
                self.ending[k] = self.script[k]["流"]

        self.plotchain_conditions = {}   # 情节链的集合
        for scene in self.scenes:
            self.plotchain_conditions[scene] = {}
            for plot in self.script[scene]["情节链"]:
                node = plot.split(" (")[0]
                condition = plot.replace(node, "").strip()
                self.plotchain_conditions[scene][node] = condition

        self.action_conditions = {}  # 对每个场景中的动作以及动作条件分离
        for scene in self.scenes:
            self.action_conditions[scene] = {}
            for interaction in self.script[scene]["交互"].get("动作选择", []):
                action = interaction.split(" (")[0]
                condition = (
                    interaction.replace(action, "")
                    .replace("收集关键线索：", "")
                    .replace("收集关键线索:", "")
                    .strip()
                )
                self.action_conditions[scene][action] = condition

        self.intention_conditions = {}
        for scene in self.scenes:
            self.intention_conditions[scene] = {}
            for interaction in self.script[scene]["交互"].get("对话", []):
                intention = interaction.split(" (")[0]
                condition = (
                    interaction.replace(intention, "")
                    .replace("收集关键线索：", "")
                    .replace("收集关键线索:", "")
                    .strip()
                )
                self.intention_conditions[scene][intention] = condition

        for scene in self.scenes:
            self.script[scene]["情节链"] = [
                x.split(" (")[0] if (" (") in x else x
                for x in self.script[scene]["情节链"]
            ]

    @property
    def llm_model(self):
        return self._llm_model

    @llm_model.setter
    def llm_model(self, value):
        self._init_llm_pool(value)

    def _init_llm_pool(self, llm_model):
        self._llm_model = llm_model
        self._llm_pool = {
            k: self._llm_model
            for k in ("nlu", "judge", "cue", "react", "prereply", "nlu_react")
        }

    async def _get_prereply(self, scene, plot, hint, show_prompt=False):
        intentions = self._get_intention(role="潘金莲")
        intentions = list(set(intentions) - set(self.interaction_history))
        if len(intentions) <= 2:
            intentions += ["玩家流露出继续当前剧情的意图"]
        prompt = PREREPLY_PROMPT_TEMP.format(
            script=str(self._get_script(scene=scene)),
            plot=plot,
            hint=hint,
            intentions=intentions,
        )
        response = await self._llm_provider.infer(
            model=self._llm_pool["prereply"],
            prompt=prompt,
            response_model=PrerepliesOutput,
        )
        if show_prompt:
            print(prompt)

        curr_prereply = {x["reply"]: x["intention"] for x in response["prereplies"]}
        self.curr_prereply = {}
        for k, v in curr_prereply.items():
            self.curr_prereply[k] = "无匹配意图"
            for intention in self.gameinfo["all_conditions"]:
                if v == intention or intention.startswith(v):
                    self.curr_prereply[k] = intention
                    break
        return response

    async def _get_judge(self, scene, plot, node, show_prompt=False):
        prompt = JUDGE_PROMPT_TEMP.format(
            script=str(self._get_script(scene=scene)),
            plot=plot,
            node=node,
        )
        response = await self._llm_provider.infer(
            model=self._llm_pool["judge"], prompt=prompt, response_model=JudgeOutput
        )
        if show_prompt:
            print(prompt)
        return response

    async def _get_cue(self, scene, plot, next_plot, show_prompt=False):
        prompt = CUE_PROMPT_TEMP.format(
            script=str(self._get_script(scene=scene)),
            plot=plot,
            next_plot=next_plot,
        )
        response = await self._llm_provider.infer(
            model=self._llm_pool["cue"], prompt=prompt, response_model=CueOutput
        )
        if show_prompt:
            print(prompt)

        dialogues = []
        for dialogue in response["dialogue"]:
            if dialogue.startswith("To") or dialogue.startswith("TO"):
                dialogues.append("潘金莲：" + dialogue.split("：")[1])
        return response

    async def _get_nlu(self, role, plot, show_prompt=False):
        intentions = self._get_intention(role=role)
        if len(intentions) == 0:
            return None

        prompt = NLU_PROMPT_TEMP.format(
            role="由玩家扮演的潘金莲" if role == "潘金莲" else role,
            plot=plot,
            intentions=intentions + ["无匹配意图"],
        )

        response = await self._llm_provider.infer(
            model=self._llm_pool["nlu"], prompt=prompt, response_model=NLUOutput
        )
        if show_prompt:
            print(prompt)
        return response

    async def _get_react(
        self,
        scene,
        plot,
        inputs,
        node,
        is_closing=False,
        emotion=None,
        show_prompt=True,
    ):
        def _format_dialogue(dialogue, role):
            dialogue = dialogue.replace(":", "：")

            if dialogue.startswith("To潘金莲："):
                dialogue = role + "：" + dialogue.split("：")[-1]
            else:
                # if "：" in dialogue:
                #     if "（" in dialogue and "）" in dialogue:
                #         dialogue = (
                #             role
                #             + dialogue.split("（")[-1].split("）")[0]
                #             + dialogue.split("：")[-1]
                #         )
                if not dialogue.startswith(role):
                    dialogue = role + "：" + dialogue

            return dialogue

        inputs = inputs.replace(":", "：")
        role, dialogue = inputs.split("To")[-1].split("：")[0], inputs.split("：")[-1]

        if is_closing:
            prompt = ClOSING_REACT_PROMPT_TEMP.format(
                script=str(self._get_script(scene=scene)),
                plot=plot,
                input=dialogue,
                role=role,
                node=node,
                emotion=emotion,
            )
        else:
            prompt = REACT_PROMPT_TEMP.format(
                script=str(self._get_script(scene=scene)),
                plot=plot,
                input=dialogue,
                role=role,
                node=node,
                emotion=emotion,
            )

        if show_prompt:
            print(prompt)

        response = await self._llm_provider.infer(
            model=self._llm_pool["react"], prompt=prompt, response_model=ReactOutput
        )

        response["dialogue"] = _format_dialogue(
            dialogue=response["dialogue"], role=role
        )

        if response["dialogue"] in self.curr_plot:
            # print("重复对话")
            response = await self._llm_provider.infer(
                model=self._llm_pool["react"], prompt=prompt, response_model=ReactOutput
            )
            response["dialogue"] = _format_dialogue(
                dialogue=response["dialogue"], role=role
            )
        return response

    async def _get_nlu_and_react(
        self, scene, plot, inputs, node, emotion=None, show_prompt=False
    ):
        inputs = inputs.replace(":", "：")
        role, dialogue = inputs.split("To")[-1].split("：")[0], inputs.split("：")[-1]

        prompt = NLU_AND_REACT_PROMPT_TEMP.format(
            script=str(self._get_script(scene=scene)),
            plot=plot,
            input=dialogue,
            role="由玩家扮演的潘金莲",
            npc=role,
            node=node,
            emotion=emotion,
            intentions=self._get_intention(role="潘金莲") + ["无匹配意图"],
        )
        response = await self._llm_provider.infer(
            model=self._llm_pool["nlu_react"],
            prompt=prompt,
            response_model=NLUandReactOutput,
        )
        if show_prompt:
            print(prompt)

        response["dialogue"] = response["dialogue"].replace(":", "：")
        if response["dialogue"].startswith("To潘金莲："):
            response["dialogue"] = role + "：" + response["dialogue"].split("：")[-1]
        else:
            if not response["dialogue"].startswith(role):
                response["dialogue"] = role + "：" + response["dialogue"]

        return response

    async def _get_relation(self, plot, roles, relation_score, show_prompt=False):
        prompt = RELATION_PROMPT_TEMP.format(
            plot=plot[1:], roles=roles, relation_score=relation_score
        )

        response = await self._llm_provider.infer(
            model=self._llm_pool["nlu"], prompt=prompt, response_model=RelationOutput
        )
        if show_prompt:
            print(prompt, response)
        return response

    def _get_intention(self, role="潘金莲", scene=None, conditions=None):
        if scene is None:
            scene = self.curr_scene
        if conditions is None:
            conditions = list(set(self.clue_history + self.interaction_history))

        intentions = []
        intention_conditions = self.intention_conditions[scene]

        for intention, condition in intention_conditions.items():
            if len(condition) == 0:
                intentions.append(intention)
                continue

            for condition_ in conditions:
                condition = condition.replace(condition_, "True")

            for condition_ in list(
                set(self.gameinfo["all_conditions"]) - set(conditions)
            ):
                condition = condition.replace(condition_, "False")

            try:
                if eval(condition):
                    intentions.append(intention)
            except Exception:
                continue

        if role == "潘金莲":
            intentions = [
                x for x in intentions if x.startswith("玩家") or x.startswith("潘金莲")
            ]
        else:
            # TODO:相近称呼如何获取对应的意图
            intentions = [x for x in intentions if x.startswith(role)]
        return intentions

    def _get_plotchain(self, scene=None, conditions=None):
        if scene is None:
            scene = self.curr_scene

        conditions = list(set(self.clue_history + self.interaction_history))
        plotchains = []
        if scene.startswith("结局"):
            return plotchains
        plotchain_conditions = self.plotchain_conditions[scene]
        for node, condition in plotchain_conditions.items():
            if len(condition) == 0:
                plotchains.append(node)
                continue

            for condition_ in conditions:
                condition = condition.replace(condition_, "True")

            for condition_ in list(
                set(self.gameinfo["all_conditions"]) - set(conditions)
            ):
                condition = condition.replace(condition_, "False")

            try:
                if eval(condition):
                    plotchains.append(node)
            except Exception:
                continue
        return plotchains

    def _get_action_space(self, scene=None, conditions=None):
        if scene is None:
            scene = self.curr_scene
        conditions = list(
            set(self.clue_history + self.interaction_history + self.plotnode_history)
        )
        action_space = []
        action_conditions = self.action_conditions[scene]
        for action, condition in action_conditions.items():
            if len(condition) == 0:
                action_space.append(action)
                continue

            for condition_ in conditions:
                condition = condition.replace(condition_, "True")

            for condition_ in list(
                set(self.gameinfo["all_conditions"]) - set(conditions)
            ):
                condition = condition.replace(condition_, "False")
            try:
                if eval(condition):
                    action_space.append(action)
            except Exception:
                continue
        if "离开$1" in action_space and "离开$1" in self.interaction_history:
            action_space = ["离开$1"] + list(
                set(action_space) - set(self.interaction_history)
            )

        else:
            action_space = list(set(action_space) - set(self.interaction_history))

        if self.curr_scene == "场景大街":
            action_space += [
                x.replace("场景", "前往") for x in self.script if "$支线" in x
            ]
        return action_space

    def _get_cutto(self, cutto_conditions, conditions=None):
        if conditions is None:
            conditions = list(set(self.clue_history + self.interaction_history))

        cuttos = []
        for cutto_condition in cutto_conditions:
            cutto = cutto_condition.split(" (")[0]
            condition = cutto_condition.replace(cutto, "").strip()
            if len(condition) == 0:
                cuttos.append(cutto)
                continue

            for condition_ in conditions:
                condition = condition.replace(condition_, "True")

            for condition_ in list(
                set(self.gameinfo["all_conditions"]) - set(conditions)
            ):
                condition = condition.replace(condition_, "False")

            try:
                if eval(condition):
                    cuttos.append(cutto)
            except Exception:
                continue
        if len(cuttos) >= 1:
            return cuttos[0]
        else:
            return None

    def _get_script(self, scene=None):
        if scene is None:
            scene = self.curr_scene

        if scene in self.script:
            script = {
                k: v
                for k, v in self.script[scene].items()
                if k in (["场景", "人物", "情节链", "流"])
            }
        script["情节链"] = self._get_plotchain()
        script["流"] = {k: v for k, v in script["流"].items() if k in script["情节链"]}

        return script

    def _get_next_plotnode(self, curr_plotnode):
        curr_plotchain = self._get_plotchain()

        if len(curr_plotchain) == 0:
            return None

        if curr_plotnode in curr_plotchain:
            curr_plotidx = curr_plotchain.index(curr_plotnode)

            if curr_plotidx >= len(curr_plotchain) - 1:
                curr_plotnode = curr_plotchain[-1]
            else:
                curr_plotnode = curr_plotchain[curr_plotidx + 1]
        else:
            curr_plotnode = curr_plotchain[-1]

        return curr_plotnode

    def _filter_dialogue(self, dialogue):
        return [
            dialogue_
            for dialogue_ in dialogue
            if not (
                dialogue_.startswith("To")
                or dialogue_.startswith("TO")
                or dialogue_.startswith("潘金莲")
            )
        ]

    def _update(
        self,
        inputs=None,
        narration=None,
        dialogue=None,
        filter_dialogue=True,
        count_interaction=True,
    ):
        if count_interaction:
            self.interaction_cnt += 1
        if inputs is not None and isinstance(inputs, str) and inputs != "":
            self.curr_plot.append("潘金莲：" + inputs.split("：")[-1])
            if self.curr_plotnode in self.plot_history[self.scene_history[-1]]:
                self.plot_history[self.scene_history[-1]][self.curr_plotnode].append(
                    "潘金莲：" + inputs.split("：")[-1]
                )
            else:
                self.plot_history[self.scene_history[-1]][self.curr_plotnode] = [
                    "潘金莲：" + inputs.split("：")[-1]
                ]

            if self.curr_plotnode in self.log[self.scene_history[-1]]:
                self.log[self.scene_history[-1]][self.curr_plotnode].append(
                    "潘金莲：" + inputs.split("：")[-1]
                )
            else:
                self.log[self.scene_history[-1]][self.curr_plotnode] = [
                    "潘金莲：" + inputs.split("：")[-1]
                ]

        if narration is not None and isinstance(narration, str) and narration != "":
            self.curr_plot.append(narration)
            if self.curr_plotnode in self.plot_history[self.scene_history[-1]]:
                self.plot_history[self.scene_history[-1]][self.curr_plotnode].append(
                    narration
                )
            else:
                self.plot_history[self.scene_history[-1]][self.curr_plotnode] = [
                    narration
                ]

            if self.curr_plotnode in self.log[self.scene_history[-1]]:
                self.log[self.scene_history[-1]][self.curr_plotnode].append(narration)
            else:
                self.log[self.scene_history[-1]][self.curr_plotnode] = [narration]

        if dialogue is not None and isinstance(dialogue, list) and len(dialogue) > 0:
            if filter_dialogue:
                dialogue = self._filter_dialogue(dialogue)
            for dialogue_ in dialogue:
                # if dialogue_ in self.curr_plot:
                # continue
                # if dialogue_.startswith("TO") or dialogue_.startswith("To"):
                #     dialogue_ = "潘金莲：" + dialogue_.split("：")[-1]
                self.curr_plot.append(dialogue_)

                if self.curr_plotnode in self.plot_history[self.scene_history[-1]]:
                    self.plot_history[self.scene_history[-1]][
                        self.curr_plotnode
                    ].append(dialogue_)
                    self.log[self.scene_history[-1]][self.curr_plotnode].append(
                        dialogue_
                    )
                else:
                    self.plot_history[self.scene_history[-1]][self.curr_plotnode] = [
                        dialogue_
                    ]
                    self.log[self.scene_history[-1]][self.curr_plotnode] = [dialogue_]

    def _to_story_info(
        self,
        story_stream={"narration": [], "dialogue": []},
        action_space=[],
        default_user_input=[],
        clues=[],
        hints=[],
        is_scene_end=False,
    ):
        return {
            "story_stream": story_stream,
            "action_space": action_space,
            "default_user_input": default_user_input,
            "clues": clues,
            "hints": hints,
            "scene_is_end": is_scene_end,
            "next_scene": self.next_scene,
            "is_game_end": True
            if self.next_scene and "结局" in self.next_scene
            else False,
        }

    def _match_trigger(self, scene, interaction):
        narration, clue, hint, cutto = None, None, None, None
        trigger = self.script[scene]["触发"]
        # print(f"trigger:{trigger} interaction:{interaction} trigger_:{trigger[interaction]}")
        if interaction not in trigger:
            return None
        trigger_ = trigger[interaction]
        if "叙事" in trigger_:
            narration = trigger_["叙事"]
        if "收集关键线索" in trigger_:
            clue = trigger_["收集关键线索"]
        if "关键提示" in trigger_:
            hint = trigger_["关键提示"]
        if "跳转" in trigger_:
            cutto = trigger_["跳转"]
            if isinstance(cutto, list):
                cutto = self._get_cutto(cutto_conditions=cutto)

        return {"narration": narration, "clue": clue, "hint": hint, "cutto": cutto}

    def _handle_trigger(self, trigger, story_stream, clues, hints):
        is_scene_end, is_game_end = False, False
        if trigger is not None:
            if trigger["narration"] is not None and trigger["narration"] != "":
                story_stream["narration"].append("[动画]" + trigger["narration"])

            if trigger["clue"] is not None:
                self.clue_history.append(trigger["clue"])
                if isinstance(trigger["clue"], str):
                    clues.append(trigger["clue"])
                elif isinstance(trigger["clue"], list):
                    clues += trigger["clue"]

            if trigger["hint"] is not None:
                self.hint_history.append(trigger["hint"])
                if isinstance(trigger["hint"], str):
                    hints.append(trigger["hint"])
                elif isinstance(trigger["hint"], list):
                    hints += trigger["hint"]

            if trigger["cutto"] is not None:
                is_scene_end = True
                self.next_scene = trigger["cutto"]
                if trigger["cutto"].startswith("结局"):
                    is_game_end = True
                    story_stream["narration"].append(
                        trigger["cutto"] + "：" + self.ending[trigger["cutto"]]
                    )
        return is_scene_end, is_game_end

    def _save_checkpoint(self, scene, plot):
        self.checkpoints[scene] = {
            "plot": plot,
            "clues": self.clue_history,
            "interactions": self.interaction_history,
        }

    def _end(self, inputs=None, story_stream=None):
        self._update(
            inputs=inputs,
            narration=story_stream["narration"],
            dialogue=story_stream["dialogue"],
        )
        return self._to_story_info(
            story_stream=story_stream,
            is_scene_end=True,
        )

    def _end_scene(
        self, story_stream={"narration": [], "dialogue": []}, clues=[], hints=[]
    ):
        self._update(
            narration=story_stream["narration"],
            dialogue=story_stream["dialogue"],
        )
        self.next_scene = None
        return self._to_story_info(
            story_stream=story_stream,
            clues=clues,
            hints=hints,
            is_scene_end=True,
            action_space=self._get_action_space(),
        )

    def _reset_scen(self, scene):
        self.scene_history.append(scene)
        self.plot_history[scene] = {}

        if scene != "场景大街" or scene not in self.log:
            self.log[scene] = {}

        self.curr_scene = scene
        self.curr_plot = []
        self.curr_relation = 50
        self.curr_plotchain = self._get_plotchain(
            scene=scene,
        )
        if len(self.curr_plotchain) > 0:
            self.curr_plotnode = self.curr_plotchain[0]
        else:
            self.curr_plotnode = None
        self.curr_prereply = {}
        self.next_scene = None
        self.interaction_cnt = -1
        self.is_garbage_time = False

    # 初始化场景
    async def init_scene(self, scene):
        self._save_checkpoint(scene=self.curr_scene, plot=self.curr_plot)
        self._reset_scen(scene=scene)

        story_stream, clues, hints = {"narration": [], "dialogue": []}, [], []

        if "$支线" in scene:
            cue = await self._get_cue(
                scene=self.curr_scene,
                plot=self.curr_plot + [self.script[self.curr_scene]["场景"]],
                next_plot=self.curr_plotnode,
                show_prompt=self._show_prompt,
            )

            story_stream["narration"].append("[小剧场]" + cue["narration"])
            story_stream["dialogue"] += cue["dialogue"]

            self._update(
                inputs=None,
                narration=cue["narration"],
                dialogue=cue["dialogue"],
                filter_dialogue=False,
            )
        else:
            dialogue = None
            if scene == "场景王婆家":
                dialogue = ["王婆：什么事？"]
            elif scene == "场景西门庆家":
                dialogue = ["西门庆：金莲，怎么了？"]
            elif scene == "场景衙门":
                dialogue = ["县令：何事？"]

            self._update(
                inputs=None,
                dialogue=dialogue,
                filter_dialogue=False,
            )

        for flow in self.script[self.curr_scene]["流"][self.curr_plotnode][:1]:
            if isinstance(flow, dict) and "关键提示" in flow:
                hints.append(flow["关键提示"])

        if scene not in ["场景大街", "序章"]:
            prereply = await self._get_prereply(
                scene=self.curr_scene,
                plot=self.curr_plot,
                hint=hints[-1] if len(hints) > 0 else "无任何提示",
                show_prompt=self._show_prompt,
            )
            default_user_input = (
                [x["reply"] for x in prereply["prereplies"]]
                if prereply is not None
                else []
            )
        else:
            default_user_input = []

        return self._to_story_info(
            story_stream=story_stream,
            action_space=self._get_action_space(),
            default_user_input=default_user_input,
            clues=clues,
            hints=hints,
            is_scene_end=False,
        )

    async def update_by_relation(self, roles=[], relation_score=50):
        relation = await self._get_relation(
            plot=self.curr_plot,
            roles=",".join(roles),
            relation_score=str(relation_score),
        )
        self.curr_relation = relation["relation"]
        return self.curr_relation

    async def update_by_user_input_1(
        self, inputs, npc_emotion="正常", use_combo_prompt=False
    ):
        inputs = inputs.replace(":", "：")
        assert not (inputs is None or inputs == "")
        is_scene_end = False
        story_stream, clues, hints = {"narration": [], "dialogue": []}, [], []
        _, dialogue = inputs.split("：")[0].replace("To", ""), inputs.split("：")[-1]

        if dialogue in self.curr_prereply:
            intention = self.curr_prereply[dialogue]
            # print("NLU-PRE-潘金莲", intention)
            react = await self._get_react(
                scene=self.curr_scene,
                plot=self.curr_plot + ["潘金莲：" + dialogue],
                inputs=inputs,
                node=self.curr_plotnode,
                emotion=npc_emotion,
                show_prompt=self._show_prompt,
            )
        else:
            if use_combo_prompt:
                task_nlu_react = self._get_nlu_and_react(
                    scene=self.curr_scene,
                    plot=self.curr_plot + ["潘金莲：" + dialogue],
                    inputs=inputs,
                    node=self.curr_plotnode,
                    emotion=npc_emotion,
                    show_prompt=self._show_prompt,
                )
                nlu_react = await task_nlu_react
                # print("NLU-and-REACT", nlu_react)
                intention = (
                    nlu_react["intention"] if nlu_react is not None else "无匹配意图"
                )
                react = {"dialogue": nlu_react["dialogue"]}

            else:
                task_nlu = self._get_nlu(
                    role="潘金莲",
                    plot=self.curr_plot + ["潘金莲：" + dialogue],
                    show_prompt=self._show_prompt,
                )
                task_react = self._get_react(
                    scene=self.curr_scene,
                    plot=self.curr_plot + ["潘金莲：" + dialogue],
                    inputs=inputs,
                    node=self.curr_plotnode,
                    emotion=npc_emotion,
                    show_prompt=self._show_prompt,
                )

                nlu, react = await asyncio.gather(task_nlu, task_react)

                # print("NLU-潘金莲", nlu)
                # print("REACT", react)
                intention = nlu["intention"] if nlu is not None else "无匹配意图"

        if intention != "无匹配意图":
            if intention not in self.interaction_history:
                self.interaction_history.append(intention)
                self.is_garbage_time = False

            trigger = self._match_trigger(scene=self.curr_scene, interaction=intention)
            is_scene_end, is_game_end = self._handle_trigger(
                trigger=trigger, story_stream=story_stream, clues=clues, hints=hints
            )

            if is_game_end:
                return self._end(inputs=inputs, story_stream=story_stream)

        if (
            self.is_garbage_time
            and self.interaction_cnt >= 0
            and random.random() >= 0.5
        ):
            react = await self._get_react(
                scene=self.curr_scene,
                plot=self.curr_plot + ["潘金莲：" + dialogue],
                inputs=inputs,
                node=self.curr_plotnode,
                emotion=npc_emotion,
                show_prompt=self._show_prompt,
                is_closing=True,
            )
            # print("REACT-GARBAGE", react)

        self._update(
            inputs=inputs,
            narration=None,
            dialogue=[react["dialogue"]],
            filter_dialogue=False if "$支线" in self.curr_scene else True,
        )

        if self.curr_plotnode in self.log[self.curr_scene]:
            self.log[self.curr_scene][self.curr_plotnode].append(intention)
        else:
            self.log[self.curr_scene][self.curr_plotnode] = [intention]

        story_stream["dialogue"] = self._filter_dialogue([react["dialogue"]])

        return self._to_story_info(
            story_stream=story_stream,
            action_space=self._get_action_space(),
            clues=clues,
            hints=hints,
            is_scene_end=is_scene_end,
        )

    async def update_by_user_input_2(self, inputs, npc_emotion="冷静"):
        inputs = inputs.replace(":", "：")
        is_scene_end = False
        story_stream, clues, hints = {"narration": [], "dialogue": []}, [], []
        role, _ = inputs.split("：")[0].replace("To", ""), inputs.split("：")[-1]

        task_nlu = self._get_nlu(
            role=role,
            plot=self.curr_plot,
            show_prompt=self._show_prompt,
        )
        task_judge = self._get_judge(
            scene=self.curr_scene,
            plot=self.curr_plot,
            node=self.curr_plotnode,
        )
        task_prereply = self._get_prereply(
            scene=self.curr_scene,
            plot=self.curr_plot,
            hint=hints[-1] if len(hints) > 0 else "无任何提示",
            show_prompt=self._show_prompt,
        )

        nlu, judge, prereply = await asyncio.gather(task_nlu, task_judge, task_prereply)

        # print("NLU-" + role, nlu)
        if nlu is not None:
            intention = nlu["intention"]
            if intention not in self.interaction_history:
                self.interaction_history.append(intention)
            trigger = self._match_trigger(scene=self.curr_scene, interaction=intention)
            is_scene_end, is_game_end = self._handle_trigger(
                trigger=trigger, story_stream=story_stream, clues=clues, hints=hints
            )
            if is_game_end:
                return self._end(story_stream=story_stream)

        # print("JUDGE", judge)
        if judge["is_comp"]:
            self.plotnode_history.append(self.curr_plotnode)
            for flow in self.script[self.curr_scene]["流"][self.curr_plotnode][1:]:
                if isinstance(flow, dict):
                    if "收集关键线索" in flow:
                        clues.append(flow["收集关键线索"])
                        self.clue_history.append(flow["收集关键线索"])

                    if "关键提示" in flow:
                        hints.append(flow["关键提示"])
                        self.hint_history.append(flow["关键提示"])

            curr_plotnode = self._get_next_plotnode(curr_plotnode=self.curr_plotnode)

            if curr_plotnode == self.curr_plotnode:
                self.is_garbage_time = True

            self.curr_plotnode = curr_plotnode
            if self.curr_plotnode is None:
                return self._end_scene(
                    story_stream=story_stream, clues=clues, hints=hints
                )

            for flow in self.script[self.curr_scene]["流"][self.curr_plotnode][:1]:
                if isinstance(flow, dict) and "关键提示" in flow:
                    hints.append(flow["关键提示"])

            self.interaction_cnt = 0
        else:
            # TODO: 对话轮数阈值
            if self.interaction_cnt >= 10:
                self.plotnode_history.append(self.curr_plotnode)
                self.curr_plotnode = self._get_next_plotnode(
                    curr_plotnode=self.curr_plotnode
                )
                self.interaction_cnt = 0

                if self.curr_plotnode is None:
                    return self._end_scene(
                        story_stream=story_stream, clues=clues, hints=hints
                    )

                for flow in self.script[self.curr_scene]["流"][self.curr_plotnode][:1]:
                    if isinstance(flow, dict) and "关键提示" in flow:
                        hints.append(flow["关键提示"])
                cue = await self._get_cue(
                    scene=self.curr_scene,
                    plot=self.curr_plot,
                    next_plot=self.curr_plotnode,
                )
                # print("CUE", cue)

                story_stream["narration"].append("[小剧场]" + cue["narration"])
                story_stream["dialogue"] += cue["dialogue"]

                self._update(
                    inputs=None,
                    narration=cue["narration"],
                    dialogue=cue["dialogue"],
                    filter_dialogue=False,
                )

        self.curr_plotchain = self._get_plotchain()
        if len(self.curr_plotchain) == 0:
            return self._end_scene(story_stream=story_stream, clues=clues, hints=hints)

        if self.curr_plotnode not in self.curr_plotchain:
            self.curr_plotnode = self.curr_plotchain[-1]
            self.is_garbage_time = True

        # print("PREREPLY", prereply)
        default_user_input = (
            [x["reply"] for x in prereply["prereplies"]] if prereply is not None else []
        )

        return self._to_story_info(
            story_stream=story_stream,
            action_space=self._get_action_space(),
            default_user_input=default_user_input,
            clues=clues,
            hints=hints,
            is_scene_end=is_scene_end,
        )

    # 动态剧本生成  
    async def gen_dynamic_script(self):
        gamelog = {}
        gamelog["plot_history"] = self.plot_history
        gamelog["clue_history"] = self.clue_history
        gamelog["hint_history"] = self.hint_history
        gamelog["interaction_history"] = self.interaction_history

        sideplot = await self._scriptwriter.gen_new_scene_script(
            script=self.script, gamelog=gamelog
        )

        dynamic_script = {
            (k.replace("_", "") + "$支线" if "结局" not in k else k.replace("_", "")): v
            for k, v in sideplot.items()
        }
        ending = {k: v for k, v in sideplot.items() if "结局" in k}

        is_checked_pass = True
        script = self.script_.copy()
        try:
            for k in dynamic_script:
                if "结局" in k:
                    continue
                if (
                    "交互" in dynamic_script[k]
                    and "动作选择" in dynamic_script[k]["交互"]
                    and "离开$1" not in dynamic_script[k]["交互"]["动作选择"]
                ):
                    dynamic_script[k]["交互"]["动作选择"].append("离开$1")

                dynamic_script[k]["触发"]["离开$1"] = {
                    "叙事": "你离开了{scene}。".format(
                        scene=k.replace("$支线", "").replace("场景", "")
                    ),
                    "跳转": "场景大街",
                }

                script["场景大街"]["触发"][k.replace("场景", "前往")] = {"跳转": k}

            for scene in script:
                if "$支线" in scene:
                    script.pop(scene)
                    script["场景大街"]["触发"].pop(scene.replace("场景", "前往"))

        except Exception as e:
            is_checked_pass = False
            print(e)
            # print(dynamic_script)

        if is_checked_pass:
            script.update(dynamic_script)
            script.update(ending)
            self._init_script(script=script)

    # 动作处理
    async def update_by_user_action(self, inputs):
        story_stream, clues, hints = {"narration": [], "dialogue": []}, [], []
        self.interaction_history.append(inputs)

        if self.curr_plotnode in self.log[self.curr_scene]:
            self.log[self.curr_scene][self.curr_plotnode].append(inputs)
        else:
            self.log[self.curr_scene][self.curr_plotnode] = [inputs]

        self.is_garbage_time = False

        trigger = self._match_trigger(scene=self.curr_scene, interaction=inputs)
        is_scene_end, is_game_end = self._handle_trigger(
            trigger=trigger, story_stream=story_stream, clues=clues, hints=hints
        )
        if is_game_end:
            return self._end(story_stream=story_stream)

        if (
            trigger is not None
            and trigger["narration"] is not None
            and trigger["narration"] != ""
        ):
            self._update(narration=trigger["narration"], count_interaction=False)

        action_space = self._get_action_space()

        self.curr_plotchain = self._get_plotchain()
        if len(self.curr_plotchain) == 0:
            return self._end_scene(story_stream=story_stream, clues=clues, hints=hints)

        if self.curr_plotnode not in self.curr_plotchain:
            self.curr_plotnode = self.curr_plotchain[-1]
            self.is_garbage_time = True

        if (
            inputs in self.script[self.curr_scene]["触发"]
            and "跳转" in self.script[self.curr_scene]["触发"][inputs]
        ):
            default_user_input = []
        else:
            prereply = await self._get_prereply(
                scene=self.curr_scene,
                plot=self.curr_plot,
                hint=hints[-1] if len(hints) > 0 else "无任何提示",
                show_prompt=self._show_prompt,
            )
            default_user_input = (
                [x["reply"] for x in prereply["prereplies"]]
                if prereply is not None
                else []
            )
            # print("PREREPLY", prereply)

        # print(self.curr_scene, "is_scene_end:", is_scene_end)

        if self._open_dynamic_script and is_scene_end and self.curr_scene != "场景大街":
            if "$支线" in self.curr_scene:
                for scene in list(self.script.keys()):
                    if "$支线" in scene:
                        self.script.pop(scene)
                        self.script["场景大街"]["触发"].pop(
                            scene.replace("场景", "前往")
                        )
                self.curr_scene = "场景大街"
            else:
                await self.gen_dynamic_script()

        return self._to_story_info(
            story_stream=story_stream,
            action_space=action_space,
            default_user_input=default_user_input,
            clues=clues,
            hints=hints,
            is_scene_end=is_scene_end,
        )
