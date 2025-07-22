import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import { message } from 'ant-design-vue'

export const useConfigStore = defineStore('config', () => {
  const config = ref({})
  const isLoading = ref(false)
  const hasError = ref(false)
  const lastUpdated = ref(null)
  const retryCount = ref(0)
  const maxRetries = 3
  const pendingRequests = ref(0)
  const abortControllers = ref([])
  const offlineMode = ref(false) // 添加离线模式标志
  
  function setConfig(newConfig) {
    if (!newConfig || typeof newConfig !== 'object') {
      console.error('无效的配置对象:', newConfig)
      return
    }
    
    config.value = newConfig
    lastUpdated.value = new Date()
    retryCount.value = 0
    hasError.value = false
    
    // 确保config里至少有基本的必要属性
    const requiredKeys = [
      'embed_model', 
      'model_provider', 
      'model_name', 
      'custom_models',
      '_config_items' // 添加_config_items作为必要属性
    ]
    
    requiredKeys.forEach(key => {
      if (config.value[key] === undefined) {
        console.warn(`配置缺少关键属性: ${key}，使用默认值`)
        
        // 设置默认值
        if (key === 'custom_models') {
          config.value[key] = [{
            custom_id: "internlm2-chat-7b",
            name: "书生浦语 InternLM2-Chat-7B",
            api_base: "http://localhost:8000",
            api_key: ""
          }]
        } else if (key === 'embed_model') {
          config.value[key] = "BAAI/bge-large-zh-v1.5"
        } else if (key === 'model_provider') {
          config.value[key] = "custom"
        } else if (key === 'model_name') {
          config.value[key] = "internlm/internlm2-chat-7b"
        } else if (key === '_config_items') {
          // 提供默认的配置项说明
          config.value[key] = {
            "embed_model": {
              "des": "向量嵌入模型",
              "choices": ["BAAI/bge-large-zh-v1.5", "BAAI/bge-small-zh", "text2vec-base-chinese"]
            },
            "reranker": {
              "des": "重排序模型",
              "choices": ["BAAI/bge-reranker-large", "BAAI/bge-reranker-base", "none"]
            },
            "enable_knowledge_base": {
              "des": "启用知识库",
              "default": true
            },
            "enable_knowledge_graph": {
              "des": "启用知识图谱",
              "default": false
            },
            "enable_web_search": {
              "des": "启用网络搜索",
              "default": false
            },
            "enable_reranker": {
              "des": "启用重排序",
              "default": true
            },
            "use_rewrite_query": {
              "des": "使用查询重写",
              "choices": ["auto", "always", "never"],
              "default": "auto"
            }
          }
        }
      }
    })
    
    // 确保model_names存在
    if (!config.value.model_names) {
      config.value.model_names = {
        "openai": {
          "name": "OpenAI API",
          "url": "https://openai.com/",
          "env": ["OPENAI_API_KEY"],
          "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        },
        "custom": {
          "name": "自定义模型",
          "url": "#",
          "env": [],
          "models": []
        }
      }
    }
    
    // 确保model_provider_status存在
    if (!config.value.model_provider_status) {
      config.value.model_provider_status = {
        "openai": false,
        "custom": true
      }
    }
  }

  async function setConfigValue(key, value) {
    let controller;
    try {
      // 离线模式下仅更新本地状态
      if (offlineMode.value) {
        config.value[key] = value;
        message.warning('离线模式：配置仅在本地更新，未同步到服务器');
        return true;
      }
      
      const oldValue = config.value[key]
      // 立即更新UI上的值
      config.value[key] = value
      
      // 创建用于此次请求的AbortController
      controller = new AbortController()
      abortControllers.value.push(controller)
      
      // 设置超时
      const timeoutId = setTimeout(() => {
        if (controller) controller.abort()
        // 失败时回滚值
        config.value[key] = oldValue
        message.error('更新配置超时，请检查网络或服务器状态')
      }, 10000) // 10秒超时
      
      const response = await fetch('/api/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          key: key,
          value: value
        }),
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        throw new Error(`更新配置失败: ${response.statusText || response.status}`)
      }
      
      const data = await response.json()
      
      if (data.code !== 200) {
        throw new Error(data.message || '服务器返回错误')
      }
      
      // 从服务器获取最新配置
      refreshConfig()
      return true
    } catch (error) {
      console.error('更新配置错误:', error)
      
      if (error.name !== 'AbortError') {
        message.error(`更新失败: ${error.message}`)
      }
      
      return false
    } finally {
      // 从数组中移除controller
      if (controller) {
        const index = abortControllers.value.indexOf(controller)
        if (index > -1) {
          abortControllers.value.splice(index, 1)
        }
      }
    }
  }

  function refreshConfig() {
    if (isLoading.value && retryCount.value > 0) {
      console.warn('已有配置请求进行中，忽略重复请求')
      return
    }
    
    // 取消所有正在进行的请求
    abortControllers.value.forEach(controller => {
      if (controller) controller.abort()
    })
    abortControllers.value = []
    
    isLoading.value = true
    hasError.value = false
    retryCount.value += 1
    pendingRequests.value = 1 // 重置为1个请求
    
    // 创建一个AbortController
    const controller = new AbortController()
    abortControllers.value.push(controller)
    
    // 设置超时
    const timeoutId = setTimeout(() => {
      if (controller) controller.abort()
      hasError.value = true
      isLoading.value = false
      message.error('获取配置超时，请检查网络或服务器状态')
    }, 15000) // 15秒超时
    
    console.log('开始获取配置...');
    
    // 实际请求
    fetch('/api/config', {
      signal: controller.signal
    })
    .then(response => {
      clearTimeout(timeoutId)
      
      if (!response.ok) {
        throw new Error(`获取配置失败: ${response.statusText || response.status}`)
      }
      return response.json()
    })
    .then(data => {
      console.log("获取配置成功:", data);
      if (data && data.code === 200 && data.data) {
        setConfig(data.data)
        offlineMode.value = false; // 成功获取配置，关闭离线模式
      } else {
        throw new Error(data.message || '获取配置返回了错误的格式或数据为空')
      }
      isLoading.value = false
      pendingRequests.value = 0
    })
    .catch(error => {
      clearTimeout(timeoutId)
      
      // 忽略abort错误的警告
      if (error.name === 'AbortError') {
        console.warn('配置请求已取消或超时')
      } else {
        console.error("获取配置错误:", error)
      }
      
      isLoading.value = false
      hasError.value = true
      pendingRequests.value = 0
      
      // 如果重试次数未达到最大值，则进行重试
      if (retryCount.value < maxRetries) {
        console.log(`重试获取配置 (${retryCount.value}/${maxRetries})...`)
        setTimeout(() => {
          refreshConfig()
        }, 1000 * retryCount.value) // 逐渐增加重试间隔
      } else {
        const errorMessage = error.name === 'AbortError' 
          ? '配置请求超时，请检查网络连接' 
          : `获取配置失败: ${error.message || '未知错误'}`
        
        message.error(errorMessage)
        retryCount.value = 0
        
        // 如果无法获取配置，启用离线模式并使用默认值
        offlineMode.value = true;
        message.warning('已切换到离线模式，部分功能可能受限');
        
        // 如果无法获取配置，使用默认值
        if (Object.keys(config.value).length === 0) {
          setConfig({
            "_config_items": {
              "embed_model": {
                "des": "向量嵌入模型",
                "choices": ["BAAI/bge-large-zh-v1.5", "BAAI/bge-small-zh", "text2vec-base-chinese"]
              },
              "reranker": {
                "des": "重排序模型",
                "choices": ["BAAI/bge-reranker-large", "BAAI/bge-reranker-base", "none"]
              },
              "enable_knowledge_base": {
                "des": "启用知识库",
                "default": true
              },
              "enable_knowledge_graph": {
                "des": "启用知识图谱",
                "default": false
              },
              "enable_web_search": {
                "des": "启用网络搜索",
                "default": false
              },
              "enable_reranker": {
                "des": "启用重排序",
                "default": true
              },
              "use_rewrite_query": {
                "des": "使用查询重写",
                "choices": ["auto", "always", "never"],
                "default": "auto"
              }
            },
            "model_names": {
              "openai": {
                "name": "OpenAI API",
                "url": "https://openai.com/",
                "env": ["OPENAI_API_KEY"],
                "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
              },
              "custom": {
                "name": "自定义模型",
                "url": "#",
                "env": [],
                "models": ["internlm2-chat-7b"]
              }
            },
            "model_provider_status": {
              "openai": false,
              "custom": true
            },
            "embed_model": "BAAI/bge-large-zh-v1.5",
            "model_provider": "custom",
            "model_name": "internlm/internlm2-chat-7b",
            "custom_models": [{
              "custom_id": "internlm2-chat-7b",
              "name": "书生浦语 InternLM2-Chat-7B",
              "api_base": "http://localhost:8000",
              "api_key": ""
            }],
            "enable_knowledge_base": true,
            "enable_knowledge_graph": false,
            "enable_web_search": false,
            "enable_reranker": true,
            "use_rewrite_query": "auto",
            "model_local_paths": {}
          })
        }
      }
    })
  }
  
  // 取消所有请求方法，用于组件销毁时调用
  function cancelAllRequests() {
    abortControllers.value.forEach(controller => {
      if (controller) controller.abort()
    })
    abortControllers.value = []
    isLoading.value = false
    pendingRequests.value = 0
  }
  
  // 切换离线模式
  function toggleOfflineMode(value) {
    offlineMode.value = value === undefined ? !offlineMode.value : value;
    if (offlineMode.value) {
      message.info('已启用离线模式，配置更改将不会同步到服务器');
    } else {
      message.info('已禁用离线模式，尝试连接服务器');
      refreshConfig();
    }
  }

  // 初始化时加载配置
  refreshConfig()

  return { 
    config, 
    isLoading, 
    hasError,
    lastUpdated,
    offlineMode,
    setConfig, 
    setConfigValue, 
    refreshConfig,
    cancelAllRequests,
    toggleOfflineMode
  }
})