# 人工智能数学原理与算法 · 课程笔记

本仓库整理《人工智能数学原理与算法基础》的学习笔记与自编实验材料。笔记从数学基础出发，依次覆盖机器学习、神经网络、图神经网络、Transformer、自监督学习和强化学习。

## 开始阅读

- [在线阅读](https://lunarkn1ght.github.io/notes/USTC-AI-Notes/)：适合在浏览器中搜索、跳转和阅读公式。
- [课程知识地图](人工智能数学原理与算法A.md)：建议从这里开始，了解各章之间的逻辑关系。
- [公共基础速查](00-公共基础/00-公共基础.md)集中收录跨章节复用的概念、算法和函数。
- [实验课](实验课/README.md)提供实验一至七的学习目标和学生入口。

章节索引使用 `status` 标记完成度：`stub` 表示只有提纲，`draft` 表示已有主体内容但仍待校订，`reviewed` 表示已完成一轮结构与内容复核。

## 目录

| 章节 | 内容 |
| --- | --- |
| 00 | [数学基础](00-数学基础/00-数学基础.md) |
| 01 | [人工智能概述](01-人工智能概述/01-人工智能概述.md) |
| 02 | [机器学习基础](02-机器学习基础/02-机器学习基础.md) |
| 03 | [神经网络基础](03-神经网络基础/03-神经网络基础.md) |
| 04 | [图神经网络](04-图神经网络/04-图神经网络.md) |
| 05 | [Transformer](05-Transformer/05-Transformer.md) |
| 06 | [自监督学习](06-自监督学习/06-自监督学习.md) |
| 07 | [强化学习](07-强化学习/07-强化学习.md) |

笔记以 Markdown 编写，正文采用 [Obsidian](https://obsidian.md/) 风格的双向链接（`[[...]]`），用 Obsidian 打开体验最佳。README 与章节入口同时提供标准 Markdown 链接，方便在 GitHub 上浏览。

## 参与贡献

欢迎提交纠错、补充解释、数值例子、参考资料和导航改进。开始前请阅读 [贡献指南](CONTRIBUTING.md)与[署名规则](ATTRIBUTION.md)，并在提交 Pull Request 前运行：

```bash
ruby scripts/check_notes.rb
```

所有贡献者保留自己贡献部分的著作权，并同意合并后的内容按 CC BY 4.0 发布。Git commit 和 Pull Request 记录作为主要贡献署名依据。

## 本地预览网站

```bash
python3 -m pip install -r requirements-docs.txt
python3 scripts/prepare_docs.py
python3 -m mkdocs serve
```

个人主页的部署流程会从本仓库 `main` 分支读取内容，并发布到 `/notes/USTC-AI-Notes/`。

## 许可

本仓库内容采用 [CC BY 4.0](LICENSE) 许可协议。转载或改编时请保留来源、许可链接并注明是否修改，具体见[署名与转载规则](ATTRIBUTION.md)。

> 笔记仅代表个人理解，如有谬误欢迎通过 Issue 或 PR 指正。

go