#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fine_tuning.py - 微调训练模块

针对水生动物疾病诊断的专用微调训练模块

作者: 蓝海智询团队
"""

import os
import json
import torch
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset, load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

from ..config.settings import get_settings
from ..utils.logger import get_logger
from .oceangpt_manager import OceanGPTManager, ModelConfig

logger = get_logger(__name__)
settings = get_settings()

@dataclass
class FineTuningConfig:
    """微调配置"""
    # 模型配置
    base_model_name: str = "OceanGPT-o-7B-v0.1"
    output_dir: str = "./saves/fine_tuned_models"
    
    # LoRA配置
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_seq_length: int = 2048
    
    # 训练控制
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    early_stopping_patience: int = 3
    
    # 数据配置
    train_test_split: float = 0.2
    seed: int = 42
    
    # 高级配置
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False

@dataclass
class DiagnosisExample:
    """疾病诊断示例"""
    animal_type: str  # 动物类型
    symptoms: List[str]  # 症状列表
    environment: Dict[str, Any]  # 环境参数
    diagnosis: str  # 诊断结果
    treatment: str  # 治疗方案
    confidence: float = 1.0  # 置信度
    source: str = "manual"  # 数据来源

class MarineDiagnosisTrainer:
    """水生动物疾病诊断训练器"""
    
    def __init__(self, config: FineTuningConfig = None):
        self.config = config or FineTuningConfig()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_data = []
        
        # 设置默认target_modules
        if self.config.lora_target_modules is None:
            self.config.lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    async def load_base_model(self):
        """加载基础模型"""
        logger.info(f"加载基础模型: {self.config.base_model_name}")
        
        try:
            # 使用OceanGPT管理器加载模型
            model_config = ModelConfig(
                model_name=self.config.base_model_name,
                device="auto",
                load_in_4bit=True
            )
            
            manager = OceanGPTManager(model_config)
            success = await manager.load_model()
            
            if not success:
                raise ValueError("基础模型加载失败")
            
            self.tokenizer = manager.tokenizer or manager.processor.tokenizer
            self.model = manager.model
            
            logger.info("基础模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载基础模型失败: {e}")
            return False
    
    def load_marine_disease_data(self, data_path: str = None) -> List[DiagnosisExample]:
        """加载水生动物疾病数据"""
        if not data_path:
            data_path = os.path.join(settings.data_dir, "marine_diseases")
        
        data_path = Path(data_path)
        examples = []
        
        try:
            # 加载不同格式的数据文件
            for file_path in data_path.glob("**/*"):
                if file_path.suffix.lower() in ['.json', '.jsonl']:
                    examples.extend(self._load_json_data(file_path))
                elif file_path.suffix.lower() in ['.csv']:
                    examples.extend(self._load_csv_data(file_path))
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    examples.extend(self._load_excel_data(file_path))
            
            logger.info(f"加载疾病诊断数据: {len(examples)} 条")
            return examples
            
        except Exception as e:
            logger.error(f"加载疾病数据失败: {e}")
            return []
    
    def _load_json_data(self, file_path: Path) -> List[DiagnosisExample]:
        """加载JSON格式数据"""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    data = [data]
                
                for item in data:
                    example = self._convert_to_diagnosis_example(item)
                    if example:
                        examples.append(example)
        
        except Exception as e:
            logger.error(f"读取JSON文件失败 {file_path}: {e}")
        
        return examples
    
    def _load_csv_data(self, file_path: Path) -> List[DiagnosisExample]:
        """加载CSV格式数据"""
        examples = []
        
        try:
            df = pd.read_csv(file_path)
            
            for _, row in df.iterrows():
                example = self._convert_to_diagnosis_example(row.to_dict())
                if example:
                    examples.append(example)
        
        except Exception as e:
            logger.error(f"读取CSV文件失败 {file_path}: {e}")
        
        return examples
    
    def _load_excel_data(self, file_path: Path) -> List[DiagnosisExample]:
        """加载Excel格式数据"""
        examples = []
        
        try:
            df = pd.read_excel(file_path)
            
            for _, row in df.iterrows():
                example = self._convert_to_diagnosis_example(row.to_dict())
                if example:
                    examples.append(example)
        
        except Exception as e:
            logger.error(f"读取Excel文件失败 {file_path}: {e}")
        
        return examples
    
    def _convert_to_diagnosis_example(self, data: dict) -> Optional[DiagnosisExample]:
        """转换数据为诊断示例格式"""
        try:
            # 支持多种数据格式
            if 'animal_type' in data and 'symptoms' in data:
                # 标准格式
                return DiagnosisExample(
                    animal_type=data.get('animal_type', ''),
                    symptoms=data.get('symptoms', []) if isinstance(data.get('symptoms'), list) 
                            else data.get('symptoms', '').split(',') if data.get('symptoms') else [],
                    environment=data.get('environment', {}) if isinstance(data.get('environment'), dict)
                               else json.loads(data.get('environment', '{}')) if data.get('environment') else {},
                    diagnosis=data.get('diagnosis', ''),
                    treatment=data.get('treatment', ''),
                    confidence=float(data.get('confidence', 1.0)),
                    source=data.get('source', 'imported')
                )
            
            elif 'instruction' in data and 'output' in data:
                # OceanInstruct格式
                instruction = data['instruction']
                output = data['output']
                
                # 尝试从指令中提取信息
                animal_type = self._extract_animal_type(instruction)
                symptoms = self._extract_symptoms(instruction)
                environment = self._extract_environment(instruction)
                
                return DiagnosisExample(
                    animal_type=animal_type,
                    symptoms=symptoms,
                    environment=environment,
                    diagnosis=output,
                    treatment=data.get('treatment', ''),
                    source='ocean_instruct'
                )
        
        except Exception as e:
            logger.warning(f"转换数据格式失败: {e}")
        
        return None
    
    def _extract_animal_type(self, text: str) -> str:
        """从文本中提取动物类型"""
        animals = ['鱼', '虾', '蟹', '贝类', '海参', '海带', '鲍鱼', '扇贝', '牡蛎']
        for animal in animals:
            if animal in text:
                return animal
        return '鱼类'  # 默认
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """从文本中提取症状"""
        symptoms_keywords = [
            '死亡', '游泳异常', '食欲不振', '体表溃烂', '鳃丝发白',
            '体色异常', '腹胀', '眼球突出', '鳞片脱落', '白点'
        ]
        
        found_symptoms = []
        for symptom in symptoms_keywords:
            if symptom in text:
                found_symptoms.append(symptom)
        
        return found_symptoms or ['未明确症状']
    
    def _extract_environment(self, text: str) -> Dict[str, Any]:
        """从文本中提取环境信息"""
        import re
        
        environment = {}
        
        # 提取温度
        temp_match = re.search(r'温度[：:]\s*(\d+(?:\.\d+)?)[°℃]?', text)
        if temp_match:
            environment['temperature'] = float(temp_match.group(1))
        
        # 提取pH值
        ph_match = re.search(r'pH[值]?[：:]\s*(\d+(?:\.\d+)?)', text)
        if ph_match:
            environment['ph'] = float(ph_match.group(1))
        
        # 提取溶氧
        do_match = re.search(r'溶氧[：:]\s*(\d+(?:\.\d+)?)', text)
        if do_match:
            environment['dissolved_oxygen'] = float(do_match.group(1))
        
        return environment
    
    def prepare_training_data(self, examples: List[DiagnosisExample]) -> Dataset:
        """准备训练数据"""
        logger.info("准备训练数据...")
        
        formatted_data = []
        
        for example in examples:
            # 构建训练文本
            prompt = self._build_diagnosis_prompt(example)
            response = self._build_diagnosis_response(example)
            
            # 组合为对话格式
            text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|end|>"
            
            formatted_data.append({
                "text": text,
                "animal_type": example.animal_type,
                "confidence": example.confidence
            })
        
        # 创建Dataset
        dataset = Dataset.from_list(formatted_data)
        
        logger.info(f"训练数据准备完成: {len(formatted_data)} 条")
        return dataset
    
    def _build_diagnosis_prompt(self, example: DiagnosisExample) -> str:
        """构建诊断提示"""
        prompt_parts = [
            f"请为以下{example.animal_type}的疾病症状进行诊断：",
            f"观察到的症状：{', '.join(example.symptoms)}"
        ]
        
        if example.environment:
            env_desc = []
            for key, value in example.environment.items():
                env_desc.append(f"{key}: {value}")
            prompt_parts.append(f"环境参数：{', '.join(env_desc)}")
        
        prompt_parts.append("请提供详细的诊断分析和治疗建议。")
        
        return "\n".join(prompt_parts)
    
    def _build_diagnosis_response(self, example: DiagnosisExample) -> str:
        """构建诊断回复"""
        response_parts = [f"诊断结果：{example.diagnosis}"]
        
        if example.treatment:
            response_parts.append(f"治疗方案：{example.treatment}")
        
        return "\n".join(response_parts)
    
    def setup_lora_config(self) -> LoraConfig:
        """设置LoRA配置"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none"
        )
    
    def data_collator(self, features):
        """数据整理函数"""
        texts = [feature["text"] for feature in features]
        
        # Tokenize
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        # 设置labels
        batch["labels"] = batch["input_ids"].clone()
        
        return batch
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        
        # 计算困惑度
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1)
        
        # 忽略padding token
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        # 计算交叉熵损失
        import torch.nn.functional as F
        loss = F.cross_entropy(torch.tensor(predictions), torch.tensor(labels))
        perplexity = torch.exp(loss).item()
        
        return {
            "perplexity": perplexity,
            "eval_loss": loss.item()
        }
    
    async def start_training(self, data_path: str = None) -> bool:
        """开始训练"""
        logger.info("🚀 开始水生动物疾病诊断模型微调训练")
        
        try:
            # 1. 加载基础模型
            if not await self.load_base_model():
                return False
            
            # 2. 加载训练数据
            examples = self.load_marine_disease_data(data_path)
            if not examples:
                logger.error("无法加载训练数据")
                return False
            
            # 3. 准备训练数据
            dataset = self.prepare_training_data(examples)
            
            # 4. 数据分割
            train_dataset, eval_dataset = self._split_dataset(dataset)
            
            # 5. 设置LoRA
            lora_config = self.setup_lora_config()
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            # 6. 设置训练参数
            training_args = self._setup_training_args()
            
            # 7. 创建训练器
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
            )
            
            # 8. 开始训练
            logger.info("开始训练...")
            train_result = self.trainer.train()
            
            # 9. 保存模型
            self.trainer.save_model()
            
            # 10. 保存训练日志
            self._save_training_log(train_result)
            
            logger.info("✅ 训练完成")
            return True
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return False
    
    def _split_dataset(self, dataset: Dataset):
        """分割数据集"""
        # 随机分割
        train_indices, eval_indices = train_test_split(
            range(len(dataset)),
            test_size=self.config.train_test_split,
            random_state=self.config.seed
        )
        
        train_dataset = dataset.select(train_indices)
        eval_dataset = dataset.select(eval_indices)
        
        logger.info(f"数据分割完成 - 训练集: {len(train_dataset)}, 验证集: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def _setup_training_args(self) -> TrainingArguments:
        """设置训练参数"""
        output_dir = os.path.join(
            self.config.output_dir,
            f"marine_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            report_to=None,  # 不使用wandb等
            run_name=f"marine_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    def _save_training_log(self, train_result):
        """保存训练日志"""
        log_data = {
            "config": asdict(self.config),
            "train_result": {
                "training_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get("train_runtime"),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
                "epoch": train_result.metrics.get("epoch")
            },
            "timestamp": datetime.now().isoformat()
        }
        
        log_file = os.path.join(self.config.output_dir, "training_log.json")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练日志已保存: {log_file}")

class MarineDiagnosisDataGenerator:
    """水生动物疾病诊断数据生成器"""
    
    def __init__(self):
        self.common_diseases = {
            "鱼类": [
                {"name": "白点病", "symptoms": ["体表白点", "游泳异常", "擦身行为"], "treatment": "使用甲基蓝药浴"},
                {"name": "烂鳃病", "symptoms": ["鳃丝发白", "呼吸困难", "食欲不振"], "treatment": "抗生素治疗"},
                {"name": "肠炎", "symptoms": ["腹胀", "粪便异常", "食欲不振"], "treatment": "调整饲料，使用益生菌"}
            ],
            "虾类": [
                {"name": "白斑病", "symptoms": ["甲壳白斑", "活力下降", "死亡率高"], "treatment": "改善水质，使用抗病毒药物"},
                {"name": "烂鳃病", "symptoms": ["鳃部发黑", "浮头", "死亡"], "treatment": "换水，使用消毒剂"}
            ]
        }
    
    def generate_training_examples(self, count: int = 1000) -> List[DiagnosisExample]:
        """生成训练示例"""
        examples = []
        
        for _ in range(count):
            # 随机选择动物类型和疾病
            animal_type = np.random.choice(list(self.common_diseases.keys()))
            disease = np.random.choice(self.common_diseases[animal_type])
            
            # 生成环境参数
            environment = self._generate_environment_params()
            
            # 创建示例
            example = DiagnosisExample(
                animal_type=animal_type,
                symptoms=disease["symptoms"],
                environment=environment,
                diagnosis=disease["name"],
                treatment=disease["treatment"],
                confidence=np.random.uniform(0.8, 1.0),
                source="generated"
            )
            
            examples.append(example)
        
        return examples
    
    def _generate_environment_params(self) -> Dict[str, Any]:
        """生成环境参数"""
        return {
            "temperature": np.random.uniform(20, 30),
            "ph": np.random.uniform(6.5, 8.5),
            "dissolved_oxygen": np.random.uniform(4, 8),
            "salinity": np.random.uniform(20, 35),
            "ammonia": np.random.uniform(0, 0.5)
        }
    
    def save_examples_to_file(self, examples: List[DiagnosisExample], file_path: str):
        """保存示例到文件"""
        data = [asdict(example) for example in examples]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已生成 {len(examples)} 个训练示例并保存到: {file_path}")

# 训练管理器
class TrainingManager:
    """训练管理器"""
    
    def __init__(self):
        self.trainer = MarineDiagnosisTrainer()
        self.data_generator = MarineDiagnosisDataGenerator()
    
    async def quick_start_training(self, use_generated_data: bool = True, generated_count: int = 1000):
        """快速开始训练"""
        logger.info("🌊 开始快速训练水生动物疾病诊断模型")
        
        try:
            # 如果使用生成数据，先生成训练数据
            if use_generated_data:
                logger.info("生成训练数据...")
                examples = self.data_generator.generate_training_examples(generated_count)
                
                # 保存生成的数据
                os.makedirs(settings.data_dir, exist_ok=True)
                data_file = os.path.join(settings.data_dir, "generated_marine_diseases.json")
                self.data_generator.save_examples_to_file(examples, data_file)
            
            # 开始训练
            success = await self.trainer.start_training()
            
            if success:
                logger.info("✅ 快速训练完成")
            else:
                logger.error("❌ 快速训练失败")
            
            return success
            
        except Exception as e:
            logger.error(f"快速训练出错: {e}")
            return False 