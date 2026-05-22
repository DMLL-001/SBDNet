# GitHub 上传文件清单 - SBDNet 网络模型

## 📋 核心网络模型文件 (必需)

### 主模型文件
- `nets/__init__.py` - 网络包初始化文件
- `nets/sbdnet.py` - **SBDNet 主模型** ⭐

### 依赖文件
- `nets/backbone.py` - MIT骨干网络 (SBDNet必需)

---

##  项目文档文件 (建议上传)

- `README.md` - 项目说明
- `.gitignore` - Git 忽略文件 (如有)
- `requirements.txt` - 依赖列表 (需新建或编辑)

---

## ❌ 不需要上传的文件

### 数据处理相关 (按你的要求排除)
- `dataset.py` - 数据集加载
- `dataloader.py` - 数据加载器
- `preprocess.py` - 数据预处理
- `utils1/dataloader.py` - 数据加载工具
- `utils1/utils_fit.py` - 训练工具 (涉及数据)
- `utils1/callbacks.py` - 回调函数

### 不相关的网络模型 (不上传)
- `nets/model.py` - Swin Transformer
- `nets/segformer.py` - SegFormer 模型
- `nets/segformer_new.py` - SegFormer 新版本
- `nets/segformer_training.py` - SegFormer 训练版本
- `nets/sbnet.py` - 其他网络变体

### 实用工具
- `train.py` - 训练脚本 (可选,可包含示例代码)
- `train_new.py` - 新训练脚本
- `test.py` - 测试脚本
- `test_new.py` - 新测试脚本
- `get_miou.py` - 指标评估
- `actions.py` - 应用相关
- `bsinet.py` - 具体实现
- `utils.py` - 通用工具
- `sbnet.py` - 其他模型变体

### 其他文件
- `main.py` - GUI应用
- `cls/MyClass.py` - GUI 类定义
- `imgs/` - 图像文件夹
- `model/` - 预训练模型和权重
- `utils1/` - 其他工具模块 (可选)

---

## 🎯 最小上传方案 (仅SBDNet模型)

**最简包含:**
```
nets/
  ├── __init__.py
  ├── sbdnet.py
  └── backbone.py
README.md
requirements.txt
.gitignore
```

**说明:** 如果要支持完整的训练和推理功能，建议也加上 `losses.py` 和 `models.py`

---

## ⭐ 最终上传方案 (SBDNet 核心模型 + Loss + Models)

**完整包含:**
```
nets/
  ├── __init__.py
  ├── sbdnet.py          # ⭐ SBDNet 主模型
  └── backbone.py        # 骨干网络 (必需依赖)

losses.py                # 损失函数
models.py                # 模型构建

README.md
requirements.txt
.gitignore
```

**说明:** 只上传与 SBDNet 相关的代码，其他网络模型文件（segformer、model、sbnet等）都不需要。

---

## 📦 requirements.txt 建议内容

```
torch>=1.9.0
numpy
```

---

## ✅ 上传前检查清单

- [ ] 确认 `nets/sbdnet.py` 中的 import 路径正确
- [ ] 检查 `nets/__init__.py` 是否需要添加导出语句
- [ ] 更新 README.md，说明如何使用模型
- [ ] 添加 requirements.txt 文件
- [ ] 移除敏感信息或模型权重文件
- [ ] 创建 .gitignore 文件 (排除权重和数据)

