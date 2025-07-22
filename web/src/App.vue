<script setup>
import { RouterLink, RouterView } from 'vue-router'
import { onMounted } from 'vue'
import { useConfigStore } from './stores/config'

const configStore = useConfigStore()

onMounted(() => {
  // 页面加载时获取配置
  configStore.refreshConfig()
})
</script>

<template>
  <div class="app-container">
    <router-view />
  </div>
</template>

<style>
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  width: 100%;
  max-width: 100vw;
  background-color: var(--background-color-global);
}

/* 设置路由视图的基础样式 */
.router-view-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  width: 100%;
}

/* 确保内容区域自动填充空间 */
.content-container {
  flex: 1;
  width: 100%;
  box-sizing: border-box;
  padding: 0;
  margin: 0 auto;
}
</style>
