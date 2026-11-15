# 分析服务接口

资料服务在 8765 端口提供 `/releases`、`/releases/term-a` 和版本内的表分页。表页携带 revision；客户端必须读取同一清单版本并验证摘要，不能合并不同版本的页。

应用服务在 8766 端口提供 `GET /catalog`、`POST /analyses`、`GET /analyses`、`GET /analyses/{id}`。当前目录登记 overview 课程资料汇总与 response_style 作答方式/可比性分析。分析实现登记在 `learning.domain.ANALYSES`，函数接收完整 tables 和 parameters，返回可序列化对象。

## response_style

`response_style` 使用同一批学员的重复问卷区分体验内容与作答方式，并判断哪些组别、哪些轮次的变化仍有可比依据。结果版本为 `response_style/1`。可调参数均有服务端范围校验，未知参数返回 422：

- `group_by`：登记分组字段列表，默认 `["site","group"]`，允许 `site`、`group`、`age_band`。
- `traits`：要分析的 trait 列表，默认全部。
- `wave_pairs`：要比较的轮次对，默认所有两两组合。
- `holdout_fraction`、`folds`、`seed`：整位学员留出比例、折数和确定性种子。
- `bootstrap`、`permutations`：区间与置换检验的重抽样次数。
- `alpha`、`reliability_threshold`、`min_pairs`、`min_items_per_direction`、`max_missing_rate`：判定阈值。

result 固定包含 `design`（含 fit/holdout 学员名单与识别假设）、`item_locations`、`reliability`、`response_style`、`response_style_levels`、`invariance`、`direction_tests`、`vignettes`、`comparability`、`changes`、`change_verdicts`、`model_selection`、`holdout`、`sensitivity`、`conclusions` 与 `limitations`。每个 trait 的 `comparability.constraints` 列出结论依赖的识别约束及通过/失败状态；`conclusions.dependencies` 汇总结论依赖的题目、锚点与约束；`sensitivity.scenarios` 报告去掉单题、只用正向/反向题、去掉某组别后变化量是否仍同号。

拟合、诊断与选型只使用 `design.fit_persons`；`design.holdout_persons` 仅在选型结束后用于 `holdout` 的一次样本外检查。可比性不成立时 `conclusions.verdict` 为 `not_supported`，`change_verdicts` 中相应变化量的 `supported` 为 false；服务不会为了得到正结论而把各组校准到同一分布。

创建请求字段为 request_key、release、analysis 和可选的 parameters 对象。request_key 是调用方的幂等编号：相同完整请求返回同一分析，相同编号不同内容返回 409。分析返回 id、request、revision 和 result。不同 request_key 是不同留存记录。缺少字段、未知分析类型或无效资料返回 422；查询不存在的记录返回 404。JSON 不允许 NaN 或 Infinity，无法估计的量使用 null 并在结果中说明原因。

保存后的分析内容不因资料更新而重算。新分析可在 result 内保存统计对象、参数、诊断及所需的版本化结构；不可覆盖其他分析或跳过资料完整性校验。数据库只保存本地分析，不是资料服务的业务真源。新增分析不改变既有 overview 或已保存记录的查询方式。
