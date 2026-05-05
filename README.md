# 意图识别与目标提取

项目只保留一套主流程：先做意图识别，再做目标提取，最终输出：

```json
{
  "action": "",
  "target": []
}
```

方案基于 `scikit-learn`，可在 CPU 上训练和推理。

建议始终使用项目内虚拟环境：

```powershell
.\.venv\Scripts\python.exe
```

当前固定依赖版本见 [requirements.txt](D:\intentionRecogination\requirements.txt)。模型训练和推理要使用同一套版本。

## 标签与对象

意图标签：

- `继续`
- `提问`
- `回答`
- `讨论`
- `举手表决`
- `下课`

固定人名：

- `李昌龙`
- `崔展豪`
- `包梓群`
- `丽娃`
- `萧华诗`
- `张晓丹`

## 业务规则

- `回答` 必须有明确 `target`
- 如果模型预测为 `回答`，但目标提取为空，则最终改成 `提问`
- 只要句子落在“提问/叫人作答”的语义里，最终只在 `提问` 和 `回答` 之间二选一
- 如果有明确作答对象，例如“张晓丹你会这道题吗”“张晓丹你来做吧”或“张晓丹你来算一算”，最终判成 `回答`，并保留该人的 `target`
- 如果只是对某个人的内容发问，而不是叫该人作答，例如“大家觉得张晓丹做得对吗”，最终判成 `提问`，返回 `target: []`
- 像“怎么解决这道题呢，我们今天学习一个方法”这类既有提问又带讲解引子的句子，最终仍判成 `提问`
- `讨论` 不做降级处理
- `下课` 固定返回 `target: []`
- `下课本` 这类包含 `下课` 子串但语义不是结束课堂的句子，不应识别成 `下课`
- 没有明确人名时，`target` 返回 `[]`

## 核心文件

- [train.py](D:\intentionRecogination\train.py)
  训练意图模型
- [target_train.py](D:\intentionRecogination\target_train.py)
  训练目标提取模型
- [target_extractor.py](D:\intentionRecogination\target_extractor.py)
  目标提取逻辑
- [pipeline.py](D:\intentionRecogination\pipeline.py)
  串联意图识别与目标提取
- [run_pipeline.py](D:\intentionRecogination\run_pipeline.py)
  统一推理入口
- [people.json](D:\intentionRecogination\people.json)
  固定人名配置

数据文件：

- [data/intent_train.example.csv](D:\intentionRecogination\data\intent_train.example.csv)
- [data/target_selector_train.example.csv](D:\intentionRecogination\data\target_selector_train.example.csv)

标注说明：

- [docs/OBJECT_LABELING_GUIDE.md](D:\intentionRecogination\docs\OBJECT_LABELING_GUIDE.md)

## 安装

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 训练

训练意图模型：

```powershell
.\.venv\Scripts\python.exe train.py --task-name intent --data data/intent_train.example.csv --output-dir models
```

训练目标模型：

```powershell
.\.venv\Scripts\python.exe target_train.py --data data/target_selector_train.example.csv --output-dir models
```

训练输出：

- `models/intent_model.joblib`
- `models/intent_metrics.json`
- `models/target_selector.joblib`
- `models/target_selector_metrics.json`

## 推理

统一推理：

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --intent-model models/intent_model.joblib --target-model models/target_selector.joblib --people-config people.json --text "张晓丹你和丽娃讨论一下这个问题。"
```

输出示例：

```json
[
  {
    "action": "讨论",
    "target": ["张晓丹", "丽娃"]
  }
]
```

没有明确人名时：

```json
[
  {
    "action": "继续",
    "target": []
  }
]
```

## 目标提取方式

目标提取采用混合方案：

1. 从 [people.json](D:\intentionRecogination\people.json) 召回候选名字
2. 用二分类模型判断候选名字是否为执行对象
3. 用少量高精度规则处理明显句式

例如：

- `崔展豪你和丽娃讨论一下这个问题。` -> `["崔展豪", "丽娃"]`
- `李昌龙你帮张晓丹看看她这一步哪里有问题。` -> `["李昌龙"]`
- `这道题其实不难，关键在于理解条件。` -> `[]`

## 部署

部署到别的项目时，通常只需要：

- [run_pipeline.py](D:\intentionRecogination\run_pipeline.py)
- [pipeline.py](D:\intentionRecogination\pipeline.py)
- [target_extractor.py](D:\intentionRecogination\target_extractor.py)
- [people.json](D:\intentionRecogination\people.json)
- `models/intent_model.joblib`
- `models/target_selector.joblib`

训练脚本和训练数据不需要随部署带走。
