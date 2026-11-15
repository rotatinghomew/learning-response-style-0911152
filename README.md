# 学习体验追踪

面向老年学习服务的内部资料分析工具。当前支持版本资料检查、学习体验追踪的描述性汇总（overview）、作答方式与跨组可比性分析（response_style）以及分析历史。它只描述已有观察，不自动生成干预结论；当证据不足时明确返回不支持比较，而不是把各组校准到同一分布。`learning/domain.py` 包含业务校验及汇总逻辑，`learning/response_style.py` 包含作答方式与可比性分析；资料源和应用是同一机器的两个本地 HTTP 进程，分别保留来源契约和分析消费契约。

## 安装和验证

使用 Python 3.11.2，数值依赖固定在 requirements.txt；部署时先安装，不在服务启动或测试过程中下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q learning scripts tests
python3 -m unittest discover -s tests -v
python3 scripts/demo.py
```

最后一条命令启动两个真实服务，等待就绪后读取目录、执行当前汇总、保存并回读结果，然后退出。它使用临时数据库，不会覆盖已有分析；服务就绪即继续，最长就绪期限为 65 秒。正常执行只需数秒。使用容器时先构建再运行，构建不计入业务演示时间：

```bash
docker build -t response-style:local .
docker run --rm response-style:local
```

## 持续运行

```bash
python3 -m learning.provider
python3 -m learning.service
```

以上命令在不同终端执行，应用默认把结果写入 var/analyses.sqlite3。数据字段、缺失含义和版本差异见 docs/data-contract.md；应用接口与幂等约定见 docs/api.md。扩展业务分析时应保持原有 overview 和已保存分析可查询。

## 作答方式与可比性

response_style 用同一批学员的重复记录把体验内容与作答方式分开：正向题与反向题等权时，平衡内容 `C=(f+r)/2` 消除共同的加性默认赞同，`S=(f-r)/2` 描述当场作答方式。它检验内容信度、逐题组别/轮次测量等价、方向负担是否随组别或轮次漂移，并把固定情境当作需要先通过顺序与收敛检验的效标，而不是天然正确的锚。结果逐组、逐轮次报告变化量与整位 bootstrap 区间，并给出结论依赖的题目和识别约束；去掉某题或某组、只用正向或反向题后是否仍成立，可在 `sensitivity` 中查到。所有拟合与选型只用 fit 学员，整位留出学员仅用于一次样本外检查。示例：

```bash
curl -s http://127.0.0.1:8766/analyses -H 'Content-Type: application/json' -d '{
  "request_key": "style-demo", "release": "term-a", "analysis": "response_style",
  "parameters": {"group_by": ["site", "group"]}
}'
```

当前 term-a 的内容信度不足且固定情境未通过检验，因此结论为不支持把“更有信心、归属感更强”写成项目总结中的变化表述。
