import logging
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from typing import Dict, List, Any, Optional, Union
import json
import os
from dataclasses import dataclass
import asyncio
import requests
from pathlib import Path

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "OceanGPT-o-7B"  # 默认使用最新的多模态版本
    model_path: str = ""
    lora_path: str = ""
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    do_sample: bool = True
    device: str = "auto"
    load_in_4bit: bool = True
    model_source: str = "huggingface"  # 默认使用 huggingface
    supports_multimodal: bool = True  # 是否支持多模态

@dataclass
class OceanGPTModels:
    """OceanGPT可用模型列表"""
    
    # 最新版本模型
    OCEANGPT_O_7B = {
        "name": "OceanGPT-o-7B",
        "base_model": "Qwen2.5-VL-7B-Instruct",
        "description": "支持多模态的海洋领域大模型",
        "supports_multimodal": True,
        "modelscope_id": "zjunlp/OceanGPT-o-7B",
        "huggingface_id": "zjunlp/OceanGPT-o-7B"
    }
    
    OCEANGPT_CODER_7B = {
        "name": "OceanGPT-coder-7B", 
        "base_model": "Qwen2.5-Coder-7B-Instruct",
        "description": "海洋领域代码生成专用模型",
        "supports_multimodal": False,
        "modelscope_id": "zjunlp/OceanGPT-coder-7B",
        "huggingface_id": "zjunlp/OceanGPT-coder-7B"
    }
    
    OCEANGPT_BASIC_V03 = {
        "name": "OceanGPT-basic-v0.3",
        "base_model": "Qwen",
        "description": "知识增强的海洋领域基础模型",
        "supports_multimodal": False,
        "modelscope_id": "zjunlp/OceanGPT-basic-v0.3",
        "huggingface_id": "zjunlp/OceanGPT-basic-v0.3"
    }
    
    # 早期版本（向后兼容）
    OCEANGPT_BASIC_14B = {
        "name": "OceanGPT-basic-14B-v0.1",
        "base_model": "Qwen1.5-14B",
        "description": "早期14B版本（效果已不如最新模型）",
        "supports_multimodal": False,
        "modelscope_id": "zjunlp/OceanGPT-basic-14B-v0.1",
        "huggingface_id": "zjunlp/OceanGPT-basic-14B-v0.1"
    }

@dataclass
class LoRAConfig:
    """LoRA微调配置"""
    r: int = 64
    lora_alpha: int = 16
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

class OceanGPTManager:
    """OceanGPT模型管理器"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.tokenizer = None
        self.model = None
        self.processor = None  # 用于多模态模型
        self.device = None
        self.model_info = None
        self._setup_device()
        self._setup_model_info()
        
    def _setup_device(self):
        """设置设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"使用CUDA设备: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                logger.info("使用CPU设备")
        else:
            self.device = self.config.device
    
    def _setup_model_info(self):
        """设置模型信息"""
        models = OceanGPTModels()
        model_map = {
            "OceanGPT-o-7B": models.OCEANGPT_O_7B,
            "OceanGPT-coder-7B": models.OCEANGPT_CODER_7B,
            "OceanGPT-basic-v0.3": models.OCEANGPT_BASIC_V03,
            "OceanGPT-basic-14B": models.OCEANGPT_BASIC_14B
        }
        
        # 支持简化的模型名称匹配
        for key, value in model_map.items():
            if key in self.config.model_name or self.config.model_name in key:
                self.model_info = value
                self.config.supports_multimodal = value["supports_multimodal"]
                break
        
        if not self.model_info:
            logger.warning(f"未识别的模型名称: {self.config.model_name}，使用默认配置")
            self.model_info = models.OCEANGPT_O_7B
    
    async def download_model_from_official(self, save_path: str = None) -> bool:
        """从官方源下载模型"""
        try:
            if not save_path:
                save_path = f"./models/{self.model_info['name']}"
            
            save_path = Path(save_path)
            if save_path.exists():
                logger.info(f"模型已存在: {save_path}")
                return True
            
            # 选择下载源
            if self.config.model_source == "modelscope":
                model_id = self.model_info["modelscope_id"]
                return await self._download_from_modelscope(model_id, save_path)
            else:
                model_id = self.model_info["huggingface_id"]
                return await self._download_from_huggingface(model_id, save_path)
        
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            return False
    
    async def _download_from_modelscope(self, model_id: str, save_path: Path) -> bool:
        """从ModelScope下载模型"""
        try:
            from modelscope import snapshot_download
            
            logger.info(f"从ModelScope下载模型: {model_id}")
            snapshot_download(
                model_id=model_id,
                cache_dir=str(save_path.parent),
                local_dir=str(save_path)
            )
            logger.info(f"模型下载完成: {save_path}")
            return True
            
        except ImportError:
            logger.error("请安装modelscope: pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"ModelScope下载失败: {e}")
            return False
    
    async def _download_from_huggingface(self, model_id: str, save_path: Path) -> bool:
        """从HuggingFace下载模型"""
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"从HuggingFace下载模型: {model_id}")
            snapshot_download(
                repo_id=model_id,
                local_dir=str(save_path),
                local_dir_use_symlinks=False
            )
            logger.info(f"模型下载完成: {save_path}")
            return True
            
        except ImportError:
            logger.error("请安装huggingface_hub: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"HuggingFace下载失败: {e}")
            return False

    async def load_model(self, model_path: str = None) -> bool:
        """加载模型"""
        try:
            model_path = model_path or self.config.model_path
            
            # 如果没有指定路径，尝试从官方下载
            if not model_path or not os.path.exists(model_path):
                logger.info("本地模型不存在，尝试从官方源下载...")
                download_success = await self.download_model_from_official(model_path)
                if not download_success:
                    raise ValueError("模型下载失败，请检查网络连接或手动下载模型")
            
            # 配置4-bit量化（仅在CUDA环境下启用）
            quantization_config = None
            if self.config.load_in_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("启用4-bit量化")
            elif self.config.load_in_4bit and self.device == "cpu":
                logger.warning("CPU环境下禁用4-bit量化")
            
            # 根据模型类型选择加载方式
            if self.config.supports_multimodal:
                return await self._load_multimodal_model(model_path, quantization_config)
            else:
                return await self._load_text_model(model_path, quantization_config)
        
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    async def _load_text_model(self, model_path: str, quantization_config) -> bool:
        """加载文本模型"""
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        # 如果有LoRA权重，加载它们
        if self.config.lora_path and os.path.exists(self.config.lora_path):
            self.model = PeftModel.from_pretrained(self.model, self.config.lora_path)
            logger.info(f"已加载LoRA权重: {self.config.lora_path}")
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        logger.info(f"文本模型加载成功: {model_path}")
        return True
    
    async def _load_multimodal_model(self, model_path: str, quantization_config) -> bool:
        """加载多模态模型"""
        try:
            # 加载processor（用于处理图像和文本）
            self.processor = Qwen2VLProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 尝试加载多模态模型，处理版本兼容性
            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
            except Exception as e:
                if "qwen2_5_vl" in str(e) or "qwen2_vl" in str(e):
                    logger.warning(f"多模态模型版本兼容性问题: {e}")
                    logger.info("尝试使用文本模式加载...")
                    # 回退到文本模式
                    return await self._load_text_model(model_path, quantization_config)
                else:
                    raise e
            
            # 如果有LoRA权重，加载它们
            if self.config.lora_path and os.path.exists(self.config.lora_path):
                self.model = PeftModel.from_pretrained(self.model, self.config.lora_path)
                logger.info(f"已加载LoRA权重: {self.config.lora_path}")
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"多模态模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"多模态模型加载失败: {e}")
            logger.info("尝试使用文本模式...")
            # 回退到文本模式
            self.config.supports_multimodal = False
            return await self._load_text_model(model_path, quantization_config)

    async def generate_response(self, prompt: str, images: List = None, **kwargs) -> str:
        """生成回复（支持文本和多模态）"""
        try:
            if not self.model:
                raise ValueError("模型尚未加载")
            
            if self.config.supports_multimodal and images:
                return await self._generate_multimodal_response(prompt, images, **kwargs)
            else:
                return await self._generate_text_response(prompt, **kwargs)
        
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return "抱歉，生成回复时出现错误。"
    
    async def _generate_text_response(self, prompt: str, **kwargs) -> str:
        """生成文本回复"""
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length - 512,
            padding=True
        )
        
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成参数
        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # 解码回复
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    async def _generate_multimodal_response(self, prompt: str, images: List, **kwargs) -> str:
        """生成多模态回复"""
        # 构建多模态输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 添加图像
        for image in images:
            messages[0]["content"].append({
                "type": "image",
                "image": image
            })
        
        # 处理输入
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 编码
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        if self.device != "cpu":
            inputs = inputs.to(self.device)
        
        # 生成参数
        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
        }
        
        # 生成回复
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_config)
        
        # 解码回复
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()

    async def fine_tune_lora(self, 
                           dataset_path: str, 
                           output_dir: str,
                           lora_config: LoRAConfig = None,
                           use_ocean_instruct: bool = True) -> bool:
        """LoRA微调"""
        try:
            if not self.model or not (self.tokenizer or self.processor):
                raise ValueError("模型尚未加载")
            
            lora_config = lora_config or LoRAConfig()
            
            # 如果启用OceanInstruct数据集，先下载
            if use_ocean_instruct:
                await self._download_ocean_instruct_data(dataset_path)
            
            # 设置LoRA配置
            if lora_config.target_modules is None:
                # 根据模型类型设置不同的target_modules
                if self.config.supports_multimodal:
                    lora_config.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                else:
                    lora_config.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias
            )
            
            # 获取PEFT模型
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            
            # 加载数据集
            train_dataset = self._load_training_dataset(dataset_path)
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                num_train_epochs=3,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                warmup_steps=100,
                logging_steps=10,
                save_steps=500,
                evaluation_strategy="no",
                save_strategy="steps",
                learning_rate=1e-4,
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                fp16=True if self.device == "cuda" else False,
            )
            
            # 创建训练器
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer or self.processor.tokenizer,
                data_collator=self._data_collator,
            )
            
            # 开始训练
            logger.info("开始LoRA微调...")
            trainer.train()
            
            # 保存模型
            trainer.save_model()
            logger.info(f"LoRA微调完成，模型保存至: {output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"LoRA微调失败: {e}")
            return False
    
    async def _download_ocean_instruct_data(self, save_path: str):
        """下载OceanInstruct指令数据"""
        try:
            # OceanInstruct数据集信息
            datasets_info = {
                "OceanInstruct-v0.2": {
                    "modelscope": "BlueOceanAI/OceanInstruct-v0.2",
                    "huggingface": "BlueOceanAI/OceanInstruct-v0.2",
                    "description": "50K中英双语海洋领域文本指令数据"
                },
                "OceanInstruct-o": {
                    "modelscope": "BlueOceanAI/OceanInstruct-o", 
                    "huggingface": "BlueOceanAI/OceanInstruct-o",
                    "description": "50K中英双语海洋领域多模态指令数据"
                }
            }
            
            # 选择合适的数据集
            dataset_name = "OceanInstruct-o" if self.config.supports_multimodal else "OceanInstruct-v0.2"
            dataset_info = datasets_info[dataset_name]
            
            logger.info(f"下载{dataset_name}数据集: {dataset_info['description']}")
            
            if self.config.model_source == "modelscope":
                await self._download_from_modelscope(dataset_info["modelscope"], Path(save_path))
            else:
                await self._download_from_huggingface(dataset_info["huggingface"], Path(save_path))
                
        except Exception as e:
            logger.warning(f"下载OceanInstruct数据失败: {e}")

    def _load_training_dataset(self, dataset_path: str):
        """加载训练数据集"""
        try:
            dataset_files = []
            
            # 支持多种数据格式
            if os.path.isdir(dataset_path):
                for file in os.listdir(dataset_path):
                    if file.endswith(('.json', '.jsonl')):
                        dataset_files.append(os.path.join(dataset_path, file))
            else:
                dataset_files.append(dataset_path)
            
            all_data = []
            for file_path in dataset_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.jsonl'):
                        data = [json.loads(line) for line in f]
                    else:
                        data = json.load(f)
                    all_data.extend(data if isinstance(data, list) else [data])
            
            # 转换为训练格式
            formatted_data = []
            for item in all_data:
                # 支持OceanInstruct格式
                if 'instruction' in item:
                    prompt = item.get('instruction', '') + '\n' + item.get('input', '')
                    response = item.get('output', '')
                elif 'conversations' in item:
                    # 支持对话格式
                    conversations = item['conversations']
                    prompt = conversations[0].get('value', '')
                    response = conversations[1].get('value', '') if len(conversations) > 1 else ''
                else:
                    continue
                
                text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|end|>"
                formatted_data.append({"text": text})
            
            logger.info(f"加载训练数据: {len(formatted_data)} 条")
            return formatted_data
            
        except Exception as e:
            logger.error(f"加载训练数据集失败: {e}")
            return []
    
    def _data_collator(self, features):
        """数据整理函数"""
        texts = [feature["text"] for feature in features]
        
        # 选择合适的tokenizer
        tokenizer = self.tokenizer or self.processor.tokenizer
        
        # Tokenize
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # 设置labels
        batch["labels"] = batch["input_ids"].clone()
        
        return batch
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        return {
            "model_name": self.config.model_name,
            "model_info": self.model_info,
            "supports_multimodal": self.config.supports_multimodal,
            "device": self.device,
            "is_loaded": self.model is not None
        }

class MarineDiseaseGPT(OceanGPTManager):
    """水生动物疾病专用GPT"""
    
    def __init__(self, config: ModelConfig = None):
        super().__init__(config)
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的水生动物疾病诊断专家。你具备以下能力：

1. 根据症状描述诊断水生动物疾病
2. 提供治疗建议和预防措施
3. 分析水质环境对疾病的影响
4. 推荐合适的药物和治疗方案

请始终：
- 提供准确、专业的医学建议
- 考虑不同水生动物的特殊性
- 强调预防的重要性
- 在不确定时建议咨询专业兽医

回答要简洁明了，重点突出，便于水产养殖从业者理解和执行。"""
    
    async def diagnose_disease(self, 
                             animal_type: str,
                             symptoms: List[str],
                             environment_info: Dict[str, Any] = None) -> str:
        """疾病诊断"""
        
        # 构建诊断提示词
        prompt_parts = [
            self.system_prompt,
            f"\n水生动物类型：{animal_type}",
            f"观察到的症状：{', '.join(symptoms)}"
        ]
        
        if environment_info:
            env_desc = []
            for key, value in environment_info.items():
                env_desc.append(f"{key}: {value}")
            prompt_parts.append(f"环境信息：{', '.join(env_desc)}")
        
        prompt_parts.append("\n请基于以上信息进行疾病诊断，并提供治疗建议：")
        
        prompt = "\n".join(prompt_parts)
        
        return await self.generate_response(prompt)
    
    async def recommend_treatment(self, disease_name: str, severity: str = "中等") -> str:
        """治疗建议"""
        prompt = f"{self.system_prompt}\n\n请为{disease_name}（严重程度：{severity}）提供详细的治疗方案和预防措施："
        
        return await self.generate_response(prompt)
    
    async def analyze_environment(self, water_parameters: Dict[str, Any]) -> str:
        """环境分析"""
        param_desc = []
        for key, value in water_parameters.items():
            param_desc.append(f"{key}: {value}")
        
        prompt = f"{self.system_prompt}\n\n请分析以下水质参数，评估对水生动物健康的影响：\n{', '.join(param_desc)}\n\n请提供水质改善建议："
        
        return await self.generate_response(prompt) 