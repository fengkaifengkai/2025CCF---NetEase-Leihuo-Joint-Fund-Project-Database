import asyncio
import json
import os
import sys
import yaml

sys.path.insert(0, sys.path[0] + "/../")

from code.scriptwriter import ScriptwriterAgent

async def test_scene_generation():
    """
    测试场景生成功能
    """
    # 加载测试剧本
    script_path = "C:\\Users\\18704\\Desktop\\aiplot-eval-master\\script\\script_PanJinLian_v2.yml"
    with open(script_path, "r", encoding="utf-8") as f:
        script = yaml.safe_load(f)
    
    # 创建测试用的玩家数据
    test_gamelog = {
        "plot_history": [
            "潘金莲毒杀了武大郎",
            "郓哥来访，潘金莲谎称武大郎生病",
            "潘金莲让郓哥离开"
        ],
        "clue_history": [
            "武大郎这两天和隔壁老王产生过巨大矛盾",
            "郓哥对潘金莲起了疑心",
            "邻居家的郓哥没有亲眼看到武大郎"
        ],
        "hint_history": [
            "要不要找王婆或者西门庆商量呢？",
            "郓哥可能发现了什么，要不要灭口呢？"
        ],
        "interaction_history": [
            {
                "scene": "序章",
                "action": "让郓哥离开$2",
                "result": "你三言两句打发了郓哥。郓哥决定离开。"
            }
        ]
    }
    
    # 创建 ScriptwriterAgent 实例
    agent = ScriptwriterAgent()
    
    # 测试场景生成
    print("开始生成新场景...")
    new_scene = await agent.gen_new_scene_script(script, test_gamelog)
    
    # 打印生成结果
    print("\n生成的新场景：")
    print(json.dumps(new_scene, ensure_ascii=False, indent=2))
    
    # 验证生成结果的格式
    print(f"dd{new_scene}")
    assert isinstance(new_scene, dict), "生成的结果必须是字典类型"
    
    # 验证场景格式
    for scene_name, scene_content in new_scene.items():
        if scene_name.startswith("场景"):
            assert "场景" in scene_content, f"场景 {scene_name} 缺少'场景'字段"
            assert "人物" in scene_content, f"场景 {scene_name} 缺少'人物'字段"
            assert "情节链" in scene_content, f"场景 {scene_name} 缺少'情节链'字段"
            assert "流" in scene_content, f"场景 {scene_name} 缺少'流'字段"
            assert "交互" in scene_content, f"场景 {scene_name} 缺少'交互'字段"
            assert "触发" in scene_content, f"场景 {scene_name} 缺少'触发'字段"
        elif scene_name.startswith("结局"):
            assert "流" in scene_content, f"结局 {scene_name} 缺少'流'字段"
    
    print("\n测试通过！生成的新场景符合格式要求。")

if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_scene_generation()) 