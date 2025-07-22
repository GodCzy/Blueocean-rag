<template>
  <div class="">
    <HeaderComponent title="设置" class="setting-header">
      <template #prefix>
        <a-button class="back-button" type="link" @click="goBack">
          <ArrowLeftOutlined /> 返回
        </a-button>
      </template>
      <template #description>
        <p>配置文件也可以在 <code>config.json</code> 中修改</p>
      </template>
      <template #actions>
        <a-button 
          :type="isNeedRestart ? 'primary' : 'default'" 
          @click="sendRestart" 
          :loading="isRestarting"
          :disabled="isRestarting || configStore.isLoading"
          :icon="h(ReloadOutlined)"
        >
          {{ isRestarting ? '重启中...' : (isNeedRestart ? '需要重启' : '重启服务') }}
        </a-button>
      </template>
    </HeaderComponent>
    
    <!-- 切换区域的标签组 (仅在移动端显示) -->
    <div class="mobile-tabs" v-if="state.windowWidth <= 520">
      <a-segmented
        :value="state.section"
        @change="(value) => state.section = value"
        :options="[
          { label: '基本设置', value: 'base' },
          { label: '模型配置', value: 'model' },
          { label: '路径配置', value: 'path' }
        ]"
      />
    </div>
    
    <!-- 加载状态 -->
    <div v-if="isLoading && !hasConfig" class="loading-container">
      <a-spin tip="正在加载配置...">
        <div class="loading-content">
          <p>首次加载可能需要一些时间，请耐心等待</p>
          <a-button v-if="hasError" type="primary" size="small" @click="configStore.refreshConfig" style="margin-top:12px">
            重试
          </a-button>
        </div>
      </a-spin>
    </div>
    
    <!-- 错误状态 -->
    <div v-else-if="hasError && !hasConfig" class="error-container">
      <a-result
        status="error"
        title="配置加载失败"
        sub-title="无法获取系统配置，请检查网络或服务器状态"
      >
        <template #extra>
          <div class="error-actions">
            <a-button type="primary" @click="configStore.refreshConfig">
              重试
            </a-button>
            <a-button @click="enableOfflineMode" :disabled="configStore.offlineMode">
              离线模式
            </a-button>
          </div>
        </template>
      </a-result>
    </div>
    
    <!-- 离线模式提示 -->
    <a-alert
      v-if="configStore.offlineMode"
      message="离线模式"
      description="当前处于离线模式，配置更改不会同步到服务器。部分功能可能受限。"
      type="warning"
      show-icon
      banner
      closable
      class="offline-banner"
    >
      <template #action>
        <a-button size="small" type="primary" @click="disableOfflineMode">
          尝试重连
        </a-button>
      </template>
    </a-alert>
    
    <!-- 正常内容 -->
    <div v-else class="setting-container">
      <div class="sider" v-if="state.windowWidth > 520">
        <a-button type="text" :class="{ activesec: state.section === 'base'}" @click="state.section='base'" :icon="h(SettingOutlined)"> 基本设置 </a-button>
        <a-button type="text" :class="{ activesec: state.section === 'model'}" @click="state.section='model'" :icon="h(CodeOutlined)"> 模型配置 </a-button>
        <a-button type="text" :class="{ activesec: state.section === 'path'}" @click="state.section='path'" :icon="h(FolderOutlined)"> 路径配置 </a-button>
      </div>
      <div class="setting" v-if="state.windowWidth <= 520 || state.section === 'base'">
        <h3>基础模型配置</h3>
        <div class="section">
          <div class="card card-select">
            <span class="label">{{ items?.embed_model.des }}</span>
            <a-select style="width: 300px"
              :value="configStore.config?.embed_model"
              @change="handleChange('embed_model', $event)"
            >
              <a-select-option
                v-for="(name, idx) in items?.embed_model.choices" :key="idx"
                :value="name">{{ name }}
              </a-select-option>
            </a-select>
          </div>
          <div class="card card-select">
            <span class="label">{{ items?.reranker.des }}</span>
            <a-select style="width: 300px"
              :value="configStore.config?.reranker"
              @change="handleChange('reranker', $event)"
              :disabled="!configStore.config.enable_reranker"
            >
              <a-select-option
                v-for="(name, idx) in items?.reranker.choices" :key="idx"
                :value="name">{{ name }}
              </a-select-option>
            </a-select>
          </div>
        </div>
        <h3>功能配置</h3>
        <div class="section">
          <div class="card">
            <span class="label">{{ items?.enable_knowledge_base.des }}</span>
            <a-switch
              :checked="configStore.config.enable_knowledge_base"
              @change="handleChange('enable_knowledge_base', !configStore.config.enable_knowledge_base)"
            />
          </div>
          <div class="card">
            <span class="label">{{ items?.enable_knowledge_graph.des }}</span>
            <a-switch
              :checked="configStore.config.enable_knowledge_graph"
              @change="handleChange('enable_knowledge_graph', !configStore.config.enable_knowledge_graph)"
            />
          </div>
          <div class="card">
            <span class="label">{{ items?.enable_web_search.des }}</span>
            <a-switch
              :checked="configStore.config.enable_web_search"
              @change="handleChange('enable_web_search', !configStore.config.enable_web_search)"
            />
          </div>
          <div class="card">
            <span class="label">{{ items?.enable_reranker.des }}</span>
            <a-switch
              :checked="configStore.config.enable_reranker"
              @change="handleChange('enable_reranker', !configStore.config.enable_reranker)"
            />
          </div>
        </div>
        <h3>检索配置</h3>
        <div class="section">
          <div class="card card-select">
            <span class="label">{{ items?.use_rewrite_query.des }}</span>
            <a-select style="width: 200px"
              :value="configStore.config?.use_rewrite_query"
              @change="handleChange('use_rewrite_query', $event)"
            >
              <a-select-option
                v-for="(name, idx) in items?.use_rewrite_query.choices" :key="idx"
                :value="name">{{ name }}
              </a-select-option>
            </a-select>
          </div>
        </div>
      </div>
      <div class="setting" v-if="state.windowWidth <= 520 || state.section === 'model'">
        <h3>模型配置</h3>
        <p>请在 <code>src/.env</code> 文件中配置对应的 APIKEY</p>
        <div class="model-provider-card">
          <div class="card-header">
            <h3>自定义模型</h3>
          </div>
          <div class="card-body">
            <div
              :class="{'model_selected': modelProvider == 'custom' && configStore.config.model_name == item.custom_id, 'card-models': true, 'custom-model': true}"
              v-for="(item, key) in configStore.config.custom_models" :key="item.custom_id"
              @click="handleChange('model_provider', 'custom'); handleChange('model_name', item.custom_id)"
            >
              <div class="card-models__header">
                <div class="name">{{ item.name }}</div>
                <div class="action">
                  <a-popconfirm
                    title="确认删除该模型?"
                    @confirm="handleDeleteCustomModel(item.custom_id)"
                    okText="确认删除"
                    cancelText="取消"
                    ok-type="danger"
                    :disabled="configStore.config.model_name == item.name"
                  >
                    <a-button type="text" :disabled="configStore.config.model_name == item.name"  @click.stop><DeleteOutlined /></a-button>
                  </a-popconfirm>
                  <a-button type="text" @click.stop="prepareToEditCustomModel(item)"><EditOutlined /></a-button>
                </div>
              </div>
              <div class="api_base">{{ item.api_base }}</div>
            </div>
            <div class="card-models custom-model" @click="prepareToAddCustomModel">
              <div class="card-models__header">
                <div class="name"> + 添加模型</div>
              </div>
              <div class="api_base">添加兼容 OpenAI 的模型</div>
              <a-modal
                class="custom-model-modal"
                v-model:open="customModel.visible"
                :title="customModel.modelTitle"
                @ok="handleAddOrEditCustomModel"
                @cancel="handleCancelCustomModel"
                :okText="'确认'"
                :cancelText="'取消'"
                :okButtonProps="{disabled: !customModel.name || !customModel.api_base}"
                :ok-type="'primary'"
              >
                <p>添加的模型是兼容 OpenAI 的模型，比如 vllm，Ollama。</p>
                <a-form :model="customModel" layout="vertical" >
                  <a-form-item label="模型名称" name="name" :rules="[{ required: true, message: '请输入模型名称' }]">
                    <a-input v-model:value="customModel.name" :disabled="customModel.edit_type == 'edit'"/>
                  </a-form-item>
                  <a-form-item label="API Base" name="api_base" :rules="[{ required: true, message: '请输入API Base' }]">
                    <a-input v-model:value="customModel.api_base" />
                  </a-form-item>
                  <a-form-item label="API KEY" name="api_key">
                    <a-input-password v-model:value="customModel.api_key" :visibilityToggle="false" autocomplete="new-password"/>
                  </a-form-item>
                </a-form>
              </a-modal>
            </div>
          </div>
        </div>
        <div class="model-provider-card" v-for="(item, key) in modelKeys" :key="key">
          <div class="card-header" @click="toggleExpand(item)">
            <!-- <div v-if="modelStatus[item]" class="success"></div> -->
            <div :class="{'model-icon': true, 'available': modelStatus[item]}">
              <img :src="modelIcons[item]" alt="模型图标">
            </div>
            <div class="model-title-container">
              <h3>{{ modelNames[item].name }}</h3>
              <a :href="modelNames[item].url" target="_blank" class="model-url" @click.stop>
                <InfoCircleOutlined />
              </a>
            </div>
            <a-button 
              type="text" 
              class="expand-button" 
              @click.stop="toggleExpand(item)"
            >
              <span class="icon-wrapper" :class="{'rotated': expandedModels[item]}">
                <DownOutlined />
              </span>
            </a-button>
          </div>
          <div class="card-body-wrapper" :class="{'expanded': expandedModels[item]}">
            <div class="card-body" v-if="modelStatus[item]">
              <div
                :class="{'model_selected': modelProvider == item && configStore.config.model_name == model, 'card-models': true}"
                v-for="(model, idx) in modelNames[item].models" :key="idx"
                @click="handleChange('model_provider', item); handleChange('model_name', model)"
              >
                <div class="model_name">{{ model }}</div>
              </div>
            </div>
          </div>
        </div>
        <div class="model-provider-card" v-for="(item, key) in notModelKeys" :key="key">
          <div class="card-header">
            <div class="model-icon">
              <img :src="modelIcons[item]" alt="模型图标">
            </div>
            <div class="model-title-container">
              <h3 style="font-weight: 400">{{ modelNames[item].name }}</h3>
              <a :href="modelNames[item].url" target="_blank" class="model-url">
                <InfoCircleOutlined />
              </a>
            </div>
            <div class="missing-keys">
              需配置<span v-for="(key, idx) in modelNames[item].env" :key="idx">{{ key }}</span>
            </div>
          </div>
        </div>
      </div>
      <div class="setting" v-if="state.windowWidth <= 520 || state.section ==='path'">
        <h3>本地模型配置</h3>
        <p>如果是 Docker 启动，务必确保在 docker-compose.dev.yaml 中添加了 volumes 映射。</p>
        <TableConfigComponent
          :config="configStore.config?.model_local_paths"
          @update:config="handleModelLocalPathsUpdate"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { message } from 'ant-design-vue';
import { computed, reactive, ref, h, watch, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router';
import { useConfigStore } from '@/stores/config';
import {
  ReloadOutlined,
  SettingOutlined,
  CodeOutlined,
  ExceptionOutlined,
  FolderOutlined,
  DeleteOutlined,
  EditOutlined,
  InfoCircleOutlined,
  DownOutlined,
  UpOutlined,
  ArrowLeftOutlined,
} from '@ant-design/icons-vue';
import HeaderComponent from '@/components/HeaderComponent.vue';
import TableConfigComponent from '@/components/TableConfigComponent.vue';
import { notification, Button } from 'ant-design-vue';
import { modelIcons } from '@/utils/modelIcon'

const router = useRouter();
const configStore = useConfigStore()
const items = computed(() => configStore.config._config_items)
const modelNames = computed(() => configStore.config?.model_names)
const modelStatus = computed(() => configStore.config?.model_provider_status)
const modelProvider = computed(() => configStore.config?.model_provider)
const isLoading = computed(() => configStore.isLoading)
const hasError = computed(() => configStore.hasError)
const hasConfig = computed(() => {
  return configStore.config && 
         Object.keys(configStore.config).length > 0 && 
         configStore.config._config_items !== undefined
})
const isNeedRestart = ref(false)
const isRestarting = ref(false)
const customModel = reactive({
  modelTitle: '添加自定义模型',
  visible: false,
  custom_id: '',
  name: '',
  api_key: '',
  api_base: '',
  edit_type: 'add',
})
const state = reactive({
  loading: false,
  section: 'base',
  windowWidth: window?.innerWidth || 0
})

// 筛选 modelStatus 中为真的key
const modelKeys = computed(() => {
  return Object.keys(modelStatus.value || {}).filter(key => modelStatus.value?.[key])
})

const notModelKeys = computed(() => {
  return Object.keys(modelStatus.value || {}).filter(key => !modelStatus.value?.[key])
})

// 模型展开状态管理
const expandedModels = reactive({})

// 监听 modelKeys 变化，确保新添加的模型也是默认展开状态
watch(modelKeys, (newKeys) => {
  newKeys.forEach(key => {
    if (expandedModels[key] === undefined) {
      expandedModels[key] = true
    }
  })
}, { immediate: true })

// 切换展开状态的方法
const toggleExpand = (item) => {
  expandedModels[item] = !expandedModels[item]
}

const generateRandomHash = (length) => {
  let chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let hash = '';
  for (let i = 0; i < length; i++) {
      hash += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return hash;
}

const handleModelLocalPathsUpdate = (config) => {
  handleChange('model_local_paths', config)
}

const handleChange = (key, e) => {
  if (key == 'enable_knowledge_graph' && e && !configStore.config.enable_knowledge_base) {
    message.error('启动知识图谱必须请先启用知识库功能')
    return
  }

  if (key == 'enable_knowledge_base' && !e && configStore.config.enable_knowledge_graph) {
    message.error('关闭知识库功能必须请先关闭知识图谱功能')
    return
  }

  // 这些都是需要重启的配置
  if (key == 'enable_reranker'
        || key == 'enable_knowledge_graph'
        || key == 'enable_knowledge_base'
        || key == 'enable_web_search'
        || key == 'embed_model'
        || key == 'reranker'
        || key == 'model_local_paths') {
    if (!isNeedRestart.value) {
      isNeedRestart.value = true
      notification.info({
        message: '需要重启服务',
        description: '请点击右下角按钮重启服务',
        placement: 'topLeft',
        duration: 0,
        btn: h(Button, { type: 'primary', onClick: sendRestart }, '立即重启')
      })
    }
  }

  configStore.setConfigValue(key, e)
}

const handleAddOrEditCustomModel = async () => {
  if (!customModel.name || !customModel.api_base) {
    message.error('请填写完整的模型名称和API Base信息。')
    return
  }

  let custom_models = configStore.config.custom_models || [];

  const model_info = {
    custom_id: customModel.custom_id || `${customModel.name}-${generateRandomHash(4)}`,
    name: customModel.name,
    api_key: customModel.api_key,
    api_base: customModel.api_base,
  }

  if (customModel.edit_type === 'add') {
    if (custom_models.find(item => item.custom_id === customModel.custom_id)) {
      message.error('模型ID已存在')
      return
    }
    custom_models.push(model_info)
  } else {
    // 如果 custom_id 相同，则更新
    custom_models = custom_models.map(item => item.custom_id === customModel.custom_id ? model_info : item);
  }

  customModel.visible = false;
  await configStore.setConfigValue('custom_models', custom_models);
  message.success(customModel.edit_type === 'add' ? '模型添加成功' : '模型修改成功');
}

const handleDeleteCustomModel = (custom_id) => {
  const updatedModels = configStore.config.custom_models.filter(item => item.custom_id !== custom_id);
  configStore.setConfigValue('custom_models', updatedModels);
}

const prepareToEditCustomModel = (item) => {
  customModel.modelTitle = '编辑自定义模型'
  customModel.custom_id = item.custom_id
  customModel.visible = true
  customModel.edit_type = 'edit'
  customModel.name = item.name
  customModel.api_key = item.api_key
  customModel.api_base = item.api_base
}

const prepareToAddCustomModel = () => {
  customModel.modelTitle = '添加自定义模型'
  customModel.edit_type = 'add'
  customModel.visible = true
  clearCustomModel()
}

const clearCustomModel = () => {
  customModel.custom_id = ''
  customModel.name = ''
  customModel.api_key = ''
  customModel.api_base = ''
}

const handleCancelCustomModel = () => {
  clearCustomModel()
  customModel.visible = false
}

const updateWindowWidth = () => {
  state.windowWidth = window?.innerWidth || 0
}

onMounted(() => {
  updateWindowWidth()
  window.addEventListener('resize', updateWindowWidth)
})

onUnmounted(() => {
  window.removeEventListener('resize', updateWindowWidth)
  configStore.cancelAllRequests()
})

const sendRestart = () => {
  console.log('正在重启服务...')
  isRestarting.value = true
  
  // 创建AbortController用于取消请求
  const controller = new AbortController()
  
  // 设置超时
  const timeoutId = setTimeout(() => {
    controller.abort()
    message.error('重启请求超时，请检查网络或服务器状态')
    isRestarting.value = false
  }, 20000) // 设置20秒超时
  
  // 请求重启
  fetch('/api/config/restart', {
    method: 'POST',
    signal: controller.signal
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`重启失败: ${response.statusText || response.status}`)
    }
    return response.json()
  })
  .then(data => {
    clearTimeout(timeoutId)
    console.log('服务已重启', data)
    message.success('服务已重启，正在重新加载配置...')
    
    // 等待服务重启后再刷新配置
    setTimeout(() => {
      configStore.refreshConfig()
      isNeedRestart.value = false
      isRestarting.value = false
      
      // 再次检查配置是否成功加载
      setTimeout(() => {
        if (configStore.hasError) {
          message.warning('重启成功，但配置加载失败，请手动刷新页面尝试重新加载')
        }
      }, 3000)
    }, 5000) // 增加等待时间从3秒到5秒，给服务更多启动时间
  })
  .catch(error => {
    clearTimeout(timeoutId)
    console.error("重启服务失败:", error)
    
    let errorMsg = '重启失败'
    if (error.name === 'AbortError') {
      errorMsg = '重启请求超时，服务可能仍在重启中'
    } else {
      errorMsg = `重启失败: ${error.message || '未知错误'}`
    }
    
    message.error(errorMsg)
    isRestarting.value = false
    
    // 尝试刷新配置，因为服务可能已经重启完成
    setTimeout(() => {
      configStore.refreshConfig()
    }, 5000)
  })
}

// 添加返回功能
const goBack = () => {
  router.back();
}

// 添加离线模式相关函数
const enableOfflineMode = () => {
  configStore.toggleOfflineMode(true);
}

const disableOfflineMode = () => {
  configStore.toggleOfflineMode(false);
}
</script>

<style lang="less" scoped>
.setting-header {
  height: auto;
  min-height: 80px;
  padding: 16px 0;
  background-color: #fff;
  border-bottom: 1px solid var(--gray-100);
  width: 100%;
  position: sticky;
  top: 0;
  z-index: 100;
}

.setting-header p {
  margin: 8px 0 0;
  color: var(--gray-500);
}

.back-button {
  margin-right: 8px;
  padding: 0 8px;
}

.mobile-tabs {
  margin: 0 16px 16px;
  text-align: center;
  padding: 16px 0 8px;
  background-color: #fff;
  border-bottom: 1px solid var(--gray-100);
  position: sticky;
  top: 80px;
  z-index: 99;
}

.setting-container {
  box-sizing: border-box;
  display: flex;
  position: relative;
  min-height: calc(100vh - var(--header-height));
  overflow: visible;
  width: 100%;
  background-color: #fff;
  flex: 1;
}

.sider {
  width: 200px;
  height: 100%;
  padding: 20px 20px 0;
  position: sticky;
  top: 100px;
  display: flex;
  flex-direction: column;
  align-items: center;
  border-right: 1px solid var(--gray-100);
  gap: 8px;
  flex-shrink: 0;
  align-self: flex-start; /* 确保侧边栏只占据需要的高度 */

  & > * {
    width: 100%;
    height: auto;
    padding: 10px 20px;
    cursor: pointer;
    transition: all 0.1s;
    text-align: left;
    font-size: 15px;
    border-radius: 6px;
    color: var(--gray-700);

    &:hover {
      background: var(--gray-50);
    }

    &.activesec {
      background: var(--gray-100);
      color: var(--gray-900);
      font-weight: 500;
    }
  }
}

.setting {
  width: 100%;
  flex: 1;
  height: 100%;
  padding: 20px 40px 80px; /* 增加底部间距，确保有足够的滚动空间 */
  overflow: visible;

  h3 {
    margin-top: 20px;
    font-weight: 600;
    color: var(--gray-900);
    font-size: 18px;
  }

  p {
    margin: 8px 0 20px;
    color: var(--gray-600);
  }

  .section {
    margin-top: 16px;
    background-color: var(--gray-50);
    padding: 20px;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    border: 1px solid var(--gray-100);
  }

  .card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background-color: white;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    min-height: 50px;
    width: 100%; /* 确保卡片占据整个宽度 */
    box-sizing: border-box;

    .label {
      font-size: 14px;
      color: var(--gray-700);
      flex: 1;
      white-space: normal;
      line-height: 1.5;
      word-break: break-word; /* 确保长文本可以换行 */
    }

    &.card-select {
      min-height: 56px;
    }
  }

  .model-provider-card {
    margin-bottom: 20px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--gray-100);
    background-color: white;
    transition: all 0.2s ease;
    width: 100%;

    &:hover {
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.07);
    }

    .card-header {
      display: flex;
      align-items: center;
      padding: 14px 20px;
      cursor: pointer;
      position: relative;
      transition: all 0.2s ease;
      
      &:hover {
        background-color: var(--gray-10);
      }

      .model-icon {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        background-color: var(--gray-50);
        overflow: hidden;
        opacity: 0.5;
        flex-shrink: 0;
        
        &.available {
          opacity: 1;
          background-color: white;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }

        img {
          width: 28px;
          height: 28px;
          object-fit: contain;
        }
      }

      .model-title-container {
        display: flex;
        align-items: center;
        flex: 1;
        min-width: 0; /* 防止文本溢出 */

        h3 {
          margin: 0;
          font-size: 16px;
          margin-right: 8px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        
        .model-url {
          color: var(--gray-500);
          font-size: 14px;
          flex-shrink: 0;

          &:hover {
            color: var(--primary-600);
          }
        }
      }

      .expand-button {
        margin-left: auto;
        flex-shrink: 0;
        
        .icon-wrapper {
          display: inline-flex;
          transition: transform 0.3s ease;
          
          &.rotated {
            transform: rotate(-180deg);
          }
        }
      }

      .missing-keys {
        margin-left: auto;
        font-size: 13px;
        color: var(--gray-500);
        text-align: right;
        max-width: 40%;
        word-break: keep-all;
        white-space: normal;
        line-height: 1.5;

        span {
          margin-left: 4px;
          color: var(--gray-700);
          
          &:not(:last-child)::after {
            content: ",";
            margin-right: 2px;
          }
        }
      }
    }

    .card-body-wrapper {
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.3s ease;
      
      &.expanded {
        max-height: 800px; /* 增加最大高度以显示更多内容 */
      }
    }

    .card-body {
      padding: 0 20px 16px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 12px;
    }
  }

  .card-models {
    padding: 12px 16px;
    border-radius: 6px;
    background-color: var(--gray-50);
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 14px;
    position: relative;
    
    &:hover {
      background-color: var(--gray-100);
    }

    &.model_selected {
      background-color: var(--primary-50);
      border: 1px solid var(--primary-200);
      
      &:hover {
        background-color: var(--primary-100);
      }
    }

    .model_name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    &.custom-model {
      .card-models__header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
        
        .name {
          font-weight: 500;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        
        .action {
          display: flex;
          gap: 4px;
          flex-shrink: 0;
        }
      }
      
      .api_base {
        font-size: 12px;
        color: var(--gray-500);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
    }
  }
}

@media (max-width: 768px) {
  .setting-container {
    flex-direction: column;
  }

  .setting {
    padding: 16px 20px 60px; /* 减少移动端的边距 */
    width: 100%;
  }

  .card.card-select {
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
    padding: 14px;

    .label {
      margin-bottom: 4px;
      font-size: 13px;
      line-height: 1.4;
    }
    
    :deep(.ant-select) {
      width: 100% !important;
    }
  }
  
  .section {
    padding: 12px !important;
  }
}

@media (max-width: 520px) {
  .setting-container {
    flex-direction: column;
  }

  .setting {
    padding: 12px 16px 40px;
    width: 100%;
  }

  .card {
    padding: 12px !important;
    width: 100%;
    box-sizing: border-box;
    
    .label {
      font-size: 13px !important;
      margin-bottom: 8px;
    }
  }

  .card.card-select {
    padding: 12px !important;
    gap: 8px;
    align-items: flex-start;
    flex-direction: column;
  }
  
  .model-provider-card .card-header {
    flex-wrap: wrap;
    padding: 12px 16px;
    
    .missing-keys {
      width: 100%;
      margin-left: 0;
      margin-top: 8px;
      max-width: 100%;
      text-align: left;
    }
    
    .model-icon {
      width: 32px;
      height: 32px;
    }
    
    .model-title-container h3 {
      font-size: 14px;
    }
  }
  
  .card-models {
    padding: 10px 12px;
  }
  
  .section {
    padding: 10px !important;
  }
}

.loading-container, .error-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
  width: 100%;
  padding: 20px;
}

.loading-content {
  min-height: 100px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin-top: 20px;
  color: var(--gray-500);
}

/* 确保卡片文本完整显示 */
.card .label {
  font-size: 14px;
  color: var(--gray-700);
  flex: 1;
  white-space: normal;
  word-break: break-word;
  overflow-wrap: break-word;
  line-height: 1.5;
}

/* 错误状态下的操作按钮 */
.error-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}

/* 离线模式提示横幅 */
.offline-banner {
  margin-bottom: 0;
}
</style>

