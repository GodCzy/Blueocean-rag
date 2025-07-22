<template>
  <div class="search-container">
    <div class="search-header">
      <h1 class="title">蓝海智询</h1>
      <p class="subtitle">基于大模型RAG知识库与知识图谱技术的水生动物疾病问答平台</p>
    </div>
    
    <div class="search-box">
      <input 
        v-model="searchQuery" 
        type="text" 
        placeholder="请输入您的问题，如：草鱼烂鳃病有什么症状？" 
        @keyup.enter="handleSearch"
      />
      <button @click="handleSearch" :disabled="isLoading">
        <span v-if="!isLoading">搜索</span>
        <span v-else>搜索中...</span>
      </button>
    </div>
    
    <div class="filter-tags" v-if="tags.length > 0">
      <span class="filter-label">快速筛选：</span>
      <div class="tag-list">
        <span 
          v-for="tag in tags" 
          :key="tag.id" 
          class="tag" 
          :class="{ active: selectedTags.includes(tag.id) }"
          @click="toggleTag(tag.id)"
        >
          {{ tag.name }}
        </span>
      </div>
    </div>

    <div class="results-container" v-if="searchResults.length > 0">
      <h2>搜索结果</h2>
      <div class="search-results">
        <div 
          v-for="(result, index) in searchResults" 
          :key="index" 
          class="result-card"
          @click="selectResult(result)"
        >
          <h3>{{ result.title }}</h3>
          <p class="result-highlight" v-html="result.highlight"></p>
          <div class="result-tags">
            <span 
              v-for="tag in result.tags" 
              :key="tag"
              class="result-tag"
            >
              {{ tag }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <div class="answer-container" v-if="answer">
      <h2>问题解答</h2>
      <div class="answer-card">
        <div class="answer-content" v-html="answer.content"></div>
        <div class="answer-sources" v-if="answer.sources && answer.sources.length > 0">
          <h4>参考资料</h4>
          <ul>
            <li v-for="(source, index) in answer.sources" :key="index">
              {{ source.title }} 
              <span class="source-relevance">相关度: {{ Math.round(source.relevance * 100) }}%</span>
            </li>
          </ul>
        </div>
      </div>
    </div>

    <div class="knowledge-graph" v-if="showGraph">
      <h2>相关知识图谱</h2>
      <div ref="graphContainer" class="graph-container"></div>
    </div>
    
    <div class="no-results" v-if="noResults">
      <p>没有找到相关结果，请尝试其他关键词或查看以下推荐内容：</p>
      <div class="recommendation-cards">
        <div 
          v-for="(rec, index) in recommendations" 
          :key="index"
          class="recommendation"
          @click="searchQuery = rec; handleSearch()"
        >
          {{ rec }}
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue';
import axios from 'axios';

export default {
  name: 'SearchInterface',
  setup() {
    const searchQuery = ref('');
    const isLoading = ref(false);
    const searchResults = ref([]);
    const answer = ref(null);
    const noResults = ref(false);
    const showGraph = ref(false);
    const graphContainer = ref(null);
    
    // 标签系统
    const tags = ref([
      { id: 'fish_species', name: '鱼类种类' },
      { id: 'disease_types', name: '疾病类型' },
      { id: 'symptoms', name: '症状表现' },
      { id: 'treatments', name: '治疗方法' },
      { id: 'medications', name: '常用药物' },
      { id: 'seasons', name: '季节流行性' }
    ]);
    
    const selectedTags = ref([]);
    
    // 推荐问题
    const recommendations = ref([
      '草鱼常见疾病有哪些？',
      '水霉病如何治疗？',
      '夏季高温如何预防鱼类疾病？',
      '烂鳃病的症状和治疗',
      '鱼类疫病的药浴方法和注意事项'
    ]);
    
    // 切换标签选择
    const toggleTag = (tagId) => {
      const index = selectedTags.value.indexOf(tagId);
      if (index === -1) {
        selectedTags.value.push(tagId);
      } else {
        selectedTags.value.splice(index, 1);
      }
      
      if (searchResults.value.length > 0) {
        handleSearch(); // 重新搜索
      }
    };
    
    // 处理搜索
    const handleSearch = async () => {
      if (!searchQuery.value.trim()) return;
      
      isLoading.value = true;
      noResults.value = false;
      answer.value = null;
      
      try {
        // 调用后端API
        const response = await axios.post('/api/search', {
          query: searchQuery.value,
          filters: selectedTags.value,
          limit: 10
        });
        
        searchResults.value = response.data.results || [];
        
        if (searchResults.value.length === 0) {
          noResults.value = true;
        } else if (searchResults.value.length === 1) {
          // 如果只有一个结果，自动选择
          selectResult(searchResults.value[0]);
        }
        
        // 检查是否有直接回答
        if (response.data.answer) {
          answer.value = response.data.answer;
        }
        
        // 检查是否有知识图谱数据
        if (response.data.graphData) {
          showGraph.value = true;
          // 在下一个tick渲染图谱
          setTimeout(() => {
            renderGraph(response.data.graphData);
          }, 100);
        } else {
          showGraph.value = false;
        }
        
      } catch (error) {
        console.error('搜索请求失败:', error);
      } finally {
        isLoading.value = false;
      }
    };
    
    // 选择搜索结果
    const selectResult = async (result) => {
      isLoading.value = true;
      
      try {
        // 获取详细信息
        const response = await axios.get(`/api/document/${result.id}`);
        
        answer.value = {
          content: response.data.content,
          sources: response.data.sources || []
        };
        
        // 获取知识图谱
        const graphResponse = await axios.get(`/api/graph/${result.id}`);
        if (graphResponse.data) {
          showGraph.value = true;
          // 在下一个tick渲染图谱
          setTimeout(() => {
            renderGraph(graphResponse.data);
          }, 100);
        }
        
      } catch (error) {
        console.error('获取详细信息失败:', error);
      } finally {
        isLoading.value = false;
      }
    };
    
    // 渲染知识图谱
    const renderGraph = (graphData) => {
      if (!graphContainer.value) return;
      
      // 这里可以使用D3.js或其他图谱库渲染
      // 简化版，实际项目中应该使用专业图谱库如ECharts、Vis.js等
      const container = graphContainer.value;
      container.innerHTML = ''; // 清空容器
      
      // 示例：使用预占位信息
      container.innerHTML = `
        <div class="graph-placeholder">
          <p>图谱数据已加载 (${graphData.nodes.length}个节点, ${graphData.edges.length}条边)</p>
          <p>实际项目中将使用D3.js或其他图谱库渲染</p>
        </div>
      `;
    };
    
    onMounted(() => {
      // 可以在这里初始化一些数据，如获取热门标签等
    });
    
    return {
      searchQuery,
      isLoading,
      searchResults,
      answer,
      tags,
      selectedTags,
      noResults,
      recommendations,
      showGraph,
      graphContainer,
      handleSearch,
      toggleTag,
      selectResult
    };
  }
}
</script>

<style scoped>
.search-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

.search-header {
  text-align: center;
  margin-bottom: 2rem;
}

.title {
  color: #0066cc;
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.subtitle {
  color: #666;
  font-size: 1rem;
}

.search-box {
  display: flex;
  margin-bottom: 1.5rem;
}

.search-box input {
  flex: 1;
  padding: 12px 20px;
  font-size: 16px;
  border: 2px solid #ddd;
  border-radius: 4px 0 0 4px;
  outline: none;
  transition: border-color 0.3s;
}

.search-box input:focus {
  border-color: #0066cc;
}

.search-box button {
  padding: 12px 24px;
  background-color: #0066cc;
  color: white;
  border: none;
  border-radius: 0 4px 4px 0;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s;
}

.search-box button:hover {
  background-color: #0052a3;
}

.search-box button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.filter-tags {
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.filter-label {
  margin-right: 1rem;
  color: #666;
}

.tag-list {
  display: flex;
  flex-wrap: wrap;
}

.tag {
  display: inline-block;
  padding: 6px 12px;
  margin: 0 8px 8px 0;
  background-color: #f0f7ff;
  color: #0066cc;
  border-radius: 16px;
  cursor: pointer;
  transition: all 0.3s;
}

.tag:hover {
  background-color: #d9e9ff;
}

.tag.active {
  background-color: #0066cc;
  color: white;
}

.results-container, .answer-container, .knowledge-graph {
  margin-top: 2rem;
}

.results-container h2, .answer-container h2, .knowledge-graph h2 {
  color: #333;
  margin-bottom: 1rem;
  border-bottom: 1px solid #eee;
  padding-bottom: 0.5rem;
}

.search-results {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
}

.result-card {
  padding: 1rem;
  border: 1px solid #eee;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.result-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.result-card h3 {
  color: #0066cc;
  margin-top: 0;
  margin-bottom: 0.5rem;
}

.result-highlight {
  color: #555;
  font-size: 0.9rem;
  margin-bottom: 1rem;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.result-highlight em {
  font-style: normal;
  background-color: #ffffc0;
  padding: 1px 2px;
}

.result-tags {
  display: flex;
  flex-wrap: wrap;
}

.result-tag {
  font-size: 0.8rem;
  padding: 2px 8px;
  margin: 0 6px 6px 0;
  background-color: #f5f5f5;
  color: #666;
  border-radius: 12px;
}

.answer-card {
  padding: 1.5rem;
  border-radius: 8px;
  background-color: #f9fcff;
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.answer-content {
  font-size: 1.1rem;
  line-height: 1.6;
  color: #333;
}

.answer-sources {
  margin-top: 1.5rem;
  padding-top: 1rem;
  border-top: 1px dashed #ddd;
}

.answer-sources h4 {
  color: #666;
  margin-bottom: 0.5rem;
}

.answer-sources ul {
  padding-left: 1.5rem;
}

.answer-sources li {
  margin-bottom: 0.3rem;
  color: #666;
}

.source-relevance {
  color: #999;
  font-size: 0.8rem;
}

.graph-container {
  height: 400px;
  background-color: #f9f9f9;
  border-radius: 8px;
  margin-top: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.graph-placeholder {
  text-align: center;
  color: #666;
}

.no-results {
  margin-top: 2rem;
  text-align: center;
}

.recommendation-cards {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 1rem;
}

.recommendation {
  background-color: #f5f7fa;
  color: #0066cc;
  padding: 0.8rem 1.2rem;
  margin: 0.5rem;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.recommendation:hover {
  background-color: #e5eefd;
}

@media (max-width: 768px) {
  .search-container {
    padding: 1rem;
  }
  
  .title {
    font-size: 2rem;
  }
  
  .search-results {
    grid-template-columns: 1fr;
  }
}
</style> 