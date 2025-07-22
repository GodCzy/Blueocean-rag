<template>
  <a-card class="config-card" style="max-width: 960px">
    <template #title>本地模型路径配置</template>
    <template #extra>
      <a-tooltip>
        <template #title>设置模型的本地路径可以使系统直接使用本地模型，避免重复下载</template>
        <a-button type="text" size="small"><QuestionCircleOutlined /></a-button>
      </a-tooltip>
    </template>
    <a-form layout="vertical">
      <div
        v-for="(item, index) in configList"
        :key="index"
        class="config-item"
      >
        <a-row :gutter="[16, 8]" align="middle">
          <a-col :xs="24" :sm="24" :md="8" :lg="8">
            <a-input
              v-model:value="item.key"
              placeholder="模型名称"
              readonly
              class="key-input"
            />
          </a-col>
          <a-col :xs="20" :sm="20" :md="14" :lg="14">
            <a-input
              v-model:value="item.value"
              @change="updateValue(index)"
              placeholder="模型本地路径"
              class="value-input"
            />
          </a-col>
          <a-col :xs="4" :sm="4" :md="2" :lg="2" class="delete-btn-col">
            <a-button
              type="link"
              danger
              class="delete-btn"
              @click="deleteConfig(index)"
            >
              <DeleteOutlined />
            </a-button>
          </a-col>
        </a-row>
      </div>
      
      <div class="empty-tip" v-if="configList.length === 0">
        <p>暂无配置，点击下方按钮添加模型路径映射</p>
      </div>

      <a-button block @click="addConfig" class="add-btn" :disabled="isAdding">
        <PlusOutlined /> 添加路径映射
      </a-button>
    </a-form>

    <a-modal
      title="添加路径映射"
      v-model:open="addConfigModalVisible"
      @ok="confirmAddConfig"
      class="config-modal"
      okText="确认"
      cancelText="取消"
      :maskClosable="false"
    >
      <a-form layout="vertical">
        <a-form-item label="模型名称（与Huggingface名称一致，例如 BAAI/bge-large-zh-v1.5）" required>
          <a-input
            v-model:value="newConfig.key"
            placeholder="请输入模型名称"
            class="modal-input"
          />
        </a-form-item>
        <a-form-item label="模型本地路径（绝对路径，例如 /path/to/models/BAAI/bge-large-zh-v1.5）" required>
          <a-input
            v-model:value="newConfig.value"
            placeholder="请输入模型本地路径"
            class="modal-input"
          />
        </a-form-item>
      </a-form>
    </a-modal>
  </a-card>
</template>

<script setup>
import { ref, reactive, computed, watch } from 'vue';
import { message } from 'ant-design-vue';
import { DeleteOutlined, PlusOutlined, QuestionCircleOutlined } from '@ant-design/icons-vue';

const props = defineProps({
  config: {
    type: Object,
    default: () => ({})
  }
});

const emit = defineEmits(['update:config']);

// 配置列表
const configList = reactive([]);

// 初始化配置列表
const initConfigList = () => {
  configList.length = 0; // 清空数组
  if (props.config) {
    Object.entries(props.config).forEach(([key, value]) => {
      configList.push({ key, value });
    });
  }
};

// 在组件创建时初始化
initConfigList();

// 当props.config变化时重新初始化
watch(() => props.config, () => {
  initConfigList();
}, { deep: true });

// 控制模态框显示
const addConfigModalVisible = ref(false);
const isAdding = ref(false); // 添加新的ref变量

// 新增配置项数据
const newConfig = ref({ key: '', value: '' });

// 添加配置
const addConfig = () => {
  isAdding.value = true;  // 设置添加状态
  newConfig.value = { key: '', value: '' }; // 重置表单
  addConfigModalVisible.value = true;
};

// 确认添加配置
const confirmAddConfig = () => {
  if (newConfig.value.key === '' || newConfig.value.value === '') {
    message.warning('键或值不能为空');
    return;
  }
  if (configList.some(item => item.key === newConfig.value.key)) {
    message.warning('键已存在');
    return;
  }
  configList.push({ key: newConfig.value.key, value: newConfig.value.value });
  addConfigModalVisible.value = false;
  newConfig.value = { key: '', value: '' };
  isAdding.value = false;  // 重置添加状态
  
  // 更新父组件数据
  emit('update:config', configObject.value);
};

// 删除配置
const deleteConfig = (index) => {
  configList.splice(index, 1);
  
  // 更新父组件数据
  emit('update:config', configObject.value);
};

// 更新值
const updateValue = (index) => {
  // 更新父组件数据
  emit('update:config', configObject.value);
};

// 将配置列表转换为对象
const configObject = computed(() => {
  return configList.reduce((acc, item) => {
    acc[item.key] = item.value;
    return acc;
  }, {});
});

// 监听模态框关闭
watch(addConfigModalVisible, (newValue) => {
  if (!newValue) {
    isAdding.value = false;  // 当模态框关闭时重置添加状态
  }
});
</script>

<style scoped>
.config-card {
  background-color: white;
  border-radius: 8px;
  border: 1px solid var(--border-color-base);
  overflow: visible;
  width: 100%;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
  margin-bottom: 24px;
}

.config-item {
  border-bottom: 1px solid var(--border-color-split);
  padding: 16px 0;
  transition: background-color 0.3s ease;
}

.config-item:hover {
  background-color: var(--gray-10);
}

.config-item:last-child {
  border-bottom: none;
}

.key-input {
  background-color: var(--gray-50);
  border-color: var(--gray-200);
  font-weight: 500;
  font-family: 'Courier New', Courier, monospace;
  width: 100%;
  white-space: normal;
  word-break: break-all;
}

.value-input {
  font-family: 'Courier New', Courier, monospace;
  width: 100%;
  max-width: none;
}

.delete-btn-col {
  display: flex;
  justify-content: center;
  align-items: center;
}

.delete-btn {
  opacity: 0.6;
  transition: opacity 0.3s ease;
}

.delete-btn:hover {
  opacity: 1;
}

.add-btn {
  margin-top: 16px;
  height: 40px;
  transition: all 0.3s ease;
  width: auto;
}

.modal-input {
  margin-bottom: 8px;
  font-family: 'Courier New', Courier, monospace;
  width: 100%;
}

.empty-tip {
  text-align: center;
  padding: 24px 0;
  color: var(--gray-500);
  background-color: var(--gray-10);
  border-radius: 6px;
  margin: 16px 0;
  border: 1px dashed var(--gray-200);
}

:deep(.ant-modal-content) {
  border-radius: 8px;
}

:deep(.ant-card-head) {
  border-bottom: 1px solid var(--gray-100);
  min-height: 56px;
}

:deep(.ant-card-head-title) {
  font-size: 16px;
  font-weight: 600;
  padding: 16px 0;
}

:deep(.ant-card-body) {
  padding: 24px;
}

:deep(.ant-form-item-label > label) {
  color: var(--gray-700);
  font-weight: 500;
  text-align: left;
  line-height: 1.4;
}

:deep(.ant-form-item-label > label.ant-form-item-required::before) {
  color: var(--error-color);
}

/* 响应式样式 */
@media (max-width: 768px) {
  .config-item {
    padding: 12px 0;
  }
  
  .key-input, .value-input {
    font-size: 14px;
    width: 100%;
  }
  
  :deep(.ant-form-item-label) {
    padding-bottom: 4px;
  }
  
  :deep(.ant-card-body) {
    padding: 16px;
  }
  
  .add-btn {
    width: 100%;
  }
  
  .empty-tip {
    padding: 16px 0;
  }
}

@media (max-width: 520px) {
  :deep(.ant-row) {
    margin-left: -8px !important;
    margin-right: -8px !important;
  }
  
  :deep(.ant-col) {
    padding-left: 8px !important;
    padding-right: 8px !important;
  }
  
  :deep(.ant-card-body) {
    padding: 12px;
  }
  
  :deep(.ant-card-head-title) {
    font-size: 15px;
    padding: 12px 0;
  }
  
  :deep(.ant-form-item-label) {
    padding-bottom: 4px;
  }
  
  .config-item {
    padding: 8px 0;
  }
  
  .add-btn {
    height: 36px;
    margin-top: 12px;
    font-size: 13px;
  }
  
  :deep(.ant-modal-title) {
    font-size: 15px;
  }
  
  :deep(.ant-form-item-label > label) {
    font-size: 13px;
  }
  
  .empty-tip {
    padding: 12px 0;
    font-size: 13px;
  }
}
</style>

