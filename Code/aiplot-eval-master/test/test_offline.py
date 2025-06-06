import sys

sys.path.insert(0, sys.path[0] + "/../")

import asyncio
import datetime
import json
import random

from lib.drama import DramaAgent


async def play(agent, inputs=None, action=None):
    # 处理用户输入的情况
    if inputs is not None:
        # 格式化输入为剧本格式：To人物：内容
        inputs = (
            "To" + agent.script[agent.curr_scene]["人物"].split("。")[0] + "：" + inputs
        )
        # 通过两种方式更新代理状态（可能需要根据不同情况处理）
        api = await agent.update_by_user_input_1(inputs)
        api = await agent.update_by_user_input_2(inputs)
    # 处理执行动作的情况
    elif action is not None:
        api = await agent.update_by_user_action(action)
    return api  # 返回更新后的代理状态

# 定义单次游戏运行的异步函数（含重试机制）
async def run_once(retry=3):
    while retry > 0:  # 重试循环
        try:
            # 加载潘金莲剧本
            agent = DramaAgent(
                script_path="..\\script\\script_PanJinLian_v2.yml",
                open_dynamic_script=True,
            )
            # 执行初始互动（用户输入"谁啊？"）
            await agent.init_scene(scene="序章")
            api = await play(agent, inputs="谁啊？", action=None)

            cnt = 0
            # 主游戏循环（最多30次交互）
            while not api["is_game_end"] and cnt <= 30:
                print("--------------------------------------")
                cnt += 1
                # 检查可用的互动方式
                if (
                    len(api["default_user_input"]) == 0
                    and len(api["action_space"]) == 0
                ):
                    if api["is_game_end"]:   # 游戏结束检查
                        break

                    if api["scene_is_end"]:  # 场景结束处理
                        print("跳转", api["next_scene"])
                        api = await agent.init_scene(scene=api["next_scene"])
                        continue
                    else:  # 无可用互动时退出
                        break
                # 决定互动类型（对话或动作）
                elif len(api["default_user_input"]) == 0:
                    interaction = "动作"
                elif len(api["action_space"]) == 0:
                    interaction = "对话"
                else:
                    # 60%概率选择对话，40%概率选择动作
                    if random.random() <= 0.6:
                        interaction = "对话"
                    else:
                        interaction = "动作"

                # 处理对话互动
                if interaction == "对话":
                    inputs = random.choice(api["default_user_input"])# 随机选择对话
                    print(interaction, inputs)
                    api = await play(agent, inputs=inputs, action=None)
                # 处理动作互动
                elif interaction == "动作":
                    # 优先处理特殊动作"离开$1"
                    if "离开$1" not in api["action_space"]:
                        action = random.choice(api["action_space"])
                    else:
                        # 20%概率选择离开，否则选其他动作
                        if random.random() <= 0.2:
                            action = "离开$1"
                        else:
                            action_space = list(
                                set(api["action_space"]) - set(["离开$1"])
                            )
                            if len(action_space) == 0:
                                action = "离开$1"
                            else:
                                action = random.choice(action_space)

                    print(interaction, action)
                    api = await play(agent, inputs=None, action=action)
                    # 场景结束处理
                    if api["is_game_end"]:
                        break
                    if api["scene_is_end"]:
                        print("跳转", api["next_scene"])
                        api = await agent.init_scene(scene=api["next_scene"])

            # 记录游戏结局
            if agent.next_scene is not None and agent.next_scene.startswith("结局"):
                agent.log["结局"] = (
                    agent.next_scene + "：" + agent.ending[agent.next_scene]
                )
            else:
                agent.log["结局"] = "游戏超时，你被当做凶手逮捕！"

            # 打印并保存游戏日志
            print(agent.log)
            log = agent.log
            with open(
                "output/gamelog_{dt}.json".format(
                    dt=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                ),
                "w",
                encoding="utf-8",
            ) as wf:
                json.dump(log, wf, ensure_ascii=False)

            break  # 成功运行后退出重试循环
        except Exception as e:
            print(retry, e)
            retry -= 1  # 出错时重试


if __name__ == "__main__":
    # 运行异步主程序
    asyncio.run(run_once(retry=1))
