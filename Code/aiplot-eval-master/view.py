import asyncio

import streamlit as st

from lib.drama import DramaAgent


async def main():
    if "agent" not in st.session_state:
        st.session_state.agent = DramaAgent(
            script_path=".\script\script_PanJinLian_v2.yml",
            open_dynamic_script=True,
        )
        st.session_state.api = await st.session_state.agent.init_scene(scene="序章")

    st.header("重生之我是潘金莲")
    st.error(
        "你是潘金莲，你毒杀了你的丈夫武大郎。这是第二天早晨，窗外阳光透过窗子洒在地面上，照出一片温暖的光斑。然而，这份温暖无法驱散你内心的阴霾。突然，门外传来一阵急促的敲门声......"
    )
    st.write(f"当前幕：{st.session_state.agent.curr_scene}")
    st.write("当前剧情：")
    st.write(st.session_state.agent.curr_plot)
    st.write("当前线索：")
    st.write(st.session_state.agent.clue_history)
    st.write("当前提示：")
    st.write(st.session_state.agent.hint_history)
    st.write("---")

    inputs = st.text_area(
        label="To"
        + st.session_state.agent.script[st.session_state.agent.curr_scene][
            "人物"
        ].split("。")[0]
        + "：",
        value="",
        max_chars=200,
        key="input_dialogue",
    )

    selected_dialogue = st.radio(
        "预置回复：",
        options=list(st.session_state.api["default_user_input"]),
        index=None,
        key="selected_dialogue",
    )

    if st.button("确认对话"):
        inputs_ = (
            "To"
            + st.session_state.agent.script[st.session_state.agent.curr_scene][
                "人物"
            ].split("。")[0]
            + "："
        )

        # 自由输入
        if inputs != "" and selected_dialogue is None:
            inputs_ = inputs_ + inputs
        else:
            inputs_ = inputs_ + selected_dialogue
        
        # 预置的回复
        st.session_state.api = await st.session_state.agent.update_by_user_input_1(
            inputs_
        )
        st.session_state.api = await st.session_state.agent.update_by_user_input_2(
            inputs_
        )
        st.rerun()

    selected_action = st.radio(
        "选择执行动作：",
        options=list(st.session_state.api["action_space"]),
        index=None,
        key="selected_action",
    )
    if st.button("确认动作"):
        # 获取被选中的动作
        if selected_action:
            st.session_state.api = await st.session_state.agent.update_by_user_action(
                selected_action
            )

            st.rerun()

    st.write("---")
    st.warning("后台日志")
    st.write(st.session_state.agent.plot_history)
    st.write(st.session_state.agent.interaction_history)
    st.write(st.session_state.api)
    st.write(st.session_state.agent.next_scene)

    if (
        st.session_state.agent.next_scene
        and st.session_state.agent.next_scene != st.session_state.agent.curr_scene
        and not st.session_state.agent.next_scene.startswith("结局")
    ):
        st.session_state.api = await st.session_state.agent.init_scene(
            scene=st.session_state.agent.next_scene
        )
        st.rerun()


if __name__ == "__main__":
    asyncio.run(main())
