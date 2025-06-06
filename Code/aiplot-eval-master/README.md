# AI剧情评估框架

AIplot-eval是一个基于AI-Native Interactive AVG （AI原生互动文字游戏）的剧情质量测评框架。其中，编剧智能体用于实时动态生成剧情（单幕/全幕），导演智能体通过剧本组织互动式游戏。

## 任务
给定全幕静态剧本（`script/script_PanJinLian_v2.yml`）和玩家历史游戏数据，动态生成高质量的单幕剧本（剧本格式需要严格按照给定剧本格式），要求剧情逻辑完整、情节连贯、内容创新。

## 目录
```python
|-- code # 该目录下代码可修改
    |-- config.py # 配置文件
    |-- llm.py # LLM Provider
    |-- scriptwriter.py # 编剧智能体（在此处实现AI剧情生成逻辑）
|-- lib # 该目录下代码不可修改
    |-- drama.py # 导演智能体
|-- output # 离线测评输出目录
|-- script
    |-- script_PanJinLian_v2.yml # 验证集剧本
|-- test
    |-- test_offline.py # 离线测评
    |-- test_online.py # 在线测评
README.md
requirements.txt
view.py # 交互式demo
```

## 使用
1. 安装和配置依赖，请参考`requirements.txt`
2. 在`code/config.py`中配置LLM Provider的`URL`和`Key`，样例只提供openailike LLM Provider实现, 其他LLM Provider请自行在`code/llm.py`中实现
3. 在`code/scriptwriter.py`中实现AI剧情生成逻辑
4. 运行`streamlit run view.py`即可启动交互式demo
5. 运行`python test/test_offline.py`即可离线测评（脚本模拟人进行交互得到游戏日志），输出结果在`output/`中
