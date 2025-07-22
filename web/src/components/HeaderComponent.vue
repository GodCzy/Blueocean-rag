<template>
  <div class="header-container">
    <div class="header-content">
      <div class="header-title">
        <div class="title-prefix" v-if="$slots.prefix">
          <slot name="prefix"></slot>
        </div>
        <div class="title-content">
          <h1>{{ title }}</h1>
          <slot name="description">
            <p v-if="description">{{ description }}</p>
          </slot>
        </div>
      </div>
      <div class="header-actions" v-if="$slots.actions">
        <slot name="actions"></slot>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    default: ''
  }
});
</script>

<style scoped lang="less">
.header-container {
  background-color: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  padding: 16px 24px;
  border-bottom: 1px solid #f0f0f0;
  position: sticky;
  top: 0;
  z-index: 1000;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-title {
  font-size: 14px;
  color: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  gap: 12px;
  
  .title-prefix {
    display: flex;
    align-items: center;
  }
  
  .title-content {
    flex: 1;
  }

  h1 {
    margin: 0;
    font-size: 20px;
    font-weight: 500;
    color: rgba(0, 0, 0, 0.85);
  }

  p {
    margin: 8px 0 0;
  }
}

.header-actions {
  display: flex;
  gap: 8px;
}

@media (max-width: 520px) {
  .header-container {
    padding: 12px 16px;
  }
  
  .header-title h1 {
    font-size: 18px;
  }
  
  .header-content {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .header-actions {
    width: 100%;
    justify-content: flex-end;
  }
}
</style>
