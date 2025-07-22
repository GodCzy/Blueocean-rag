<template>
  <div class="knowledge-graph-container">
    <div class="graph-controls" v-if="showControls">
      <div class="control-group">
        <label>节点大小：</label>
        <input 
          type="range" 
          min="20" 
          max="80" 
          v-model="nodeSize" 
          @input="updateGraph" 
        />
      </div>
      <div class="control-group">
        <label>关系强度：</label>
        <input 
          type="range" 
          min="1" 
          max="10" 
          v-model="linkStrength" 
          @input="updateGraph" 
        />
      </div>
      <div class="filter-section">
        <h4>节点类型筛选：</h4>
        <div class="node-type-filters">
          <label 
            v-for="(color, type) in nodeColors" 
            :key="type" 
            :style="{ borderColor: color }"
            class="node-type-option"
          >
            <input 
              type="checkbox" 
              :value="type" 
              v-model="activeNodeTypes" 
              @change="filterNodes"
            />
            <span>{{ formatNodeType(type) }}</span>
          </label>
        </div>
      </div>
    </div>
    
    <div class="graph-header">
      <h3 v-if="title">{{ title }}</h3>
      <div class="graph-actions">
        <button @click="toggleControls" class="action-button">
          {{ showControls ? '隐藏控制项' : '显示控制项' }}
        </button>
        <button @click="resetGraph" class="action-button">重置视图</button>
        <button @click="exportGraph" class="action-button">导出图谱</button>
      </div>
    </div>
    
    <div class="graph-viewport">
      <div ref="graphContainer" class="graph-container"></div>
      
      <!-- 图例 -->
      <div class="graph-legend" v-if="showLegend">
        <h4>图例</h4>
        <div 
          v-for="(color, type) in nodeColors" 
          :key="type"
          class="legend-item"
        >
          <span 
            class="legend-color" 
            :style="{ backgroundColor: color }"
          ></span>
          <span class="legend-label">{{ formatNodeType(type) }}</span>
        </div>
      </div>
      
      <!-- 节点详情 -->
      <div v-if="selectedNode" class="node-details">
        <div class="node-header">
          <h4>{{ selectedNode.name }}</h4>
          <span 
            class="node-type-badge"
            :style="{ backgroundColor: nodeColors[selectedNode.type] }"
          >
            {{ formatNodeType(selectedNode.type) }}
          </span>
          <button @click="selectedNode = null" class="close-button">×</button>
        </div>
        
        <div class="node-properties">
          <div 
            v-for="(value, key) in selectedNode.properties" 
            :key="key"
            class="property-item"
          >
            <strong>{{ formatPropertyName(key) }}：</strong>
            <span>{{ value }}</span>
          </div>
        </div>
        
        <div class="node-relations" v-if="selectedNodeRelations.length > 0">
          <h5>相关联的节点</h5>
          <div 
            v-for="(relation, index) in selectedNodeRelations" 
            :key="index"
            class="relation-item"
            @click="selectNodeById(relation.targetId)"
          >
            <span class="relation-name">{{ relation.relationName }}</span>
            <span class="target-node">{{ relation.targetName }}</span>
            <span 
              class="target-node-type"
              :style="{ backgroundColor: nodeColors[relation.targetType] }"
            >
              {{ formatNodeType(relation.targetType) }}
            </span>
          </div>
        </div>
      </div>
    </div>
    
    <div class="info-panel" v-if="!selectedNode && graphData">
      <p>共 {{ graphData.nodes.length }} 个节点，{{ graphData.edges.length }} 条关系</p>
      <p v-if="centerNode">中心节点：{{ centerNode.name }}</p>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted, watch, computed } from 'vue';
import * as d3 from 'd3';

export default {
  name: 'KnowledgeGraph',
  props: {
    graphData: {
      type: Object,
      default: () => ({ nodes: [], edges: [] })
    },
    title: {
      type: String,
      default: '疾病知识图谱'
    },
    width: {
      type: Number,
      default: 1000
    },
    height: {
      type: Number,
      default: 600
    }
  },
  emits: ['nodeClick', 'edgeClick'],
  setup(props, { emit }) {
    // 基础状态
    const graphContainer = ref(null);
    const showControls = ref(false);
    const showLegend = ref(true);
    const selectedNode = ref(null);
    const centerNode = ref(null);
    const simulation = ref(null);
    const svg = ref(null);
    
    // 图形配置
    const nodeSize = ref(40);
    const linkStrength = ref(5);
    const nodeColors = ref({
      'disease': '#E74C3C',
      'fish_species': '#2ECC71',
      'symptom': '#3498DB',
      'treatment': '#9B59B6',
      'medication': '#F1C40F',
      'environment': '#1ABC9C',
      'season': '#E67E22',
      'pathogen': '#34495E'
    });
    
    // 筛选配置
    const activeNodeTypes = ref(Object.keys(nodeColors.value));
    
    // 计算属性：选中节点的关联关系
    const selectedNodeRelations = computed(() => {
      if (!selectedNode.value || !props.graphData) return [];
      
      const nodeId = selectedNode.value.id;
      const relations = [];
      
      // 查找与当前节点相关的所有边
      props.graphData.edges.forEach(edge => {
        if (edge.source === nodeId) {
          // 查找目标节点
          const targetNode = props.graphData.nodes.find(n => n.id === edge.target);
          if (targetNode) {
            relations.push({
              relationName: edge.label,
              targetId: targetNode.id,
              targetName: targetNode.name,
              targetType: targetNode.type
            });
          }
        } else if (edge.target === nodeId) {
          // 查找源节点
          const sourceNode = props.graphData.nodes.find(n => n.id === edge.source);
          if (sourceNode) {
            relations.push({
              relationName: `被${edge.label}`,
              targetId: sourceNode.id,
              targetName: sourceNode.name,
              targetType: sourceNode.type
            });
          }
        }
      });
      
      return relations;
    });
    
    // 方法：格式化节点类型名称
    const formatNodeType = (type) => {
      const typeMap = {
        'disease': '疾病',
        'fish_species': '鱼类',
        'symptom': '症状',
        'treatment': '治疗方法',
        'medication': '药物',
        'environment': '环境因素',
        'season': '季节', 
        'pathogen': '病原体'
      };
      
      return typeMap[type] || type;
    };
    
    // 方法：格式化属性名称
    const formatPropertyName = (name) => {
      const nameMap = {
        'name': '名称',
        'type': '类型',
        'description': '描述',
        'severity': '严重程度',
        'prevalence': '流行程度',
        'infectivity': '传染性',
        'mortality': '死亡率'
      };
      
      return nameMap[name] || name;
    };
    
    // 方法：切换控制面板显示
    const toggleControls = () => {
      showControls.value = !showControls.value;
    };
    
    // 方法：重置图谱视图
    const resetGraph = () => {
      if (simulation.value) {
        simulation.value.alpha(1).restart();
      }
    };
    
    // 方法：导出图谱为PNG
    const exportGraph = () => {
      if (!svg.value) return;
      
      // 获取SVG数据
      const svgData = new XMLSerializer().serializeToString(svg.value.node());
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      
      // 创建图像
      const image = new Image();
      image.onload = function() {
        canvas.width = props.width;
        canvas.height = props.height;
        context.drawImage(image, 0, 0);
        
        // 下载
        const a = document.createElement('a');
        a.download = `${props.title || '知识图谱'}_${new Date().toISOString().slice(0, 10)}.png`;
        a.href = canvas.toDataURL('image/png');
        a.click();
      };
      
      // 转换SVG为图像
      image.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
    };
    
    // 方法：根据ID选择节点
    const selectNodeById = (nodeId) => {
      const node = props.graphData.nodes.find(n => n.id === nodeId);
      if (node) {
        selectedNode.value = node;
      }
    };
    
    // 方法：根据节点类型筛选显示
    const filterNodes = () => {
      if (!svg.value) return;
      
      // 更新节点可见性
      svg.value.selectAll('.node')
        .style('display', d => activeNodeTypes.value.includes(d.type) ? 'block' : 'none');
      
      // 更新连线可见性：两端节点都可见时连线才可见
      svg.value.selectAll('.link')
        .style('display', d => {
          const sourceNode = props.graphData.nodes.find(n => n.id === d.source.id || n.id === d.source);
          const targetNode = props.graphData.nodes.find(n => n.id === d.target.id || n.id === d.target);
          
          if (!sourceNode || !targetNode) return 'none';
          
          return activeNodeTypes.value.includes(sourceNode.type) && 
                 activeNodeTypes.value.includes(targetNode.type) ? 'block' : 'none';
        });
    };
    
    // 方法：更新图谱配置
    const updateGraph = () => {
      if (!svg.value) return;
      
      // 更新节点大小
      svg.value.selectAll('.node circle')
        .attr('r', d => {
          // 中心节点略大
          if (centerNode.value && d.id === centerNode.value.id) {
            return parseInt(nodeSize.value) * 1.3;
          }
          return nodeSize.value;
        });
      
      // 更新连线强度
      if (simulation.value) {
        simulation.value
          .force('link')
          .strength(linkStrength.value / 10);
        
        simulation.value.alpha(0.3).restart();
      }
    };
    
    // 初始化图谱
    const initGraph = () => {
      if (!graphContainer.value || !props.graphData || 
          !props.graphData.nodes || !props.graphData.nodes.length) {
        return;
      }
      
      // 清除之前的图谱
      d3.select(graphContainer.value).selectAll('*').remove();
      
      // 创建SVG元素
      svg.value = d3.select(graphContainer.value)
        .append('svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${props.width} ${props.height}`)
        .call(d3.zoom()
          .scaleExtent([0.1, 4])
          .on('zoom', (event) => {
            mainGroup.attr('transform', event.transform);
          }));
      
      // 主要内容组
      const mainGroup = svg.value.append('g');
      
      // 处理节点和连线数据
      const nodes = JSON.parse(JSON.stringify(props.graphData.nodes));
      const links = JSON.parse(JSON.stringify(props.graphData.edges)).map(link => ({
        source: link.source,
        target: link.target,
        label: link.label,
        value: link.value || 1
      }));
      
      // 寻找中心节点（通常是关系最多的节点）
      const nodeDegrees = {};
      links.forEach(link => {
        nodeDegrees[link.source] = (nodeDegrees[link.source] || 0) + 1;
        nodeDegrees[link.target] = (nodeDegrees[link.target] || 0) + 1;
      });
      
      let maxDegree = 0;
      let maxDegreeNodeId = null;
      for (const [nodeId, degree] of Object.entries(nodeDegrees)) {
        if (degree > maxDegree) {
          maxDegree = degree;
          maxDegreeNodeId = nodeId;
        }
      }
      
      if (maxDegreeNodeId) {
        centerNode.value = nodes.find(n => n.id === maxDegreeNodeId);
      }
      
      // 创建力导向模拟
      simulation.value = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links)
          .id(d => d.id)
          .distance(100)
          .strength(linkStrength.value / 10))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(props.width / 2, props.height / 2))
        .force('collision', d3.forceCollide().radius(nodeSize.value * 1.5));
      
      // 绘制连线
      const link = mainGroup.append('g')
        .attr('class', 'links')
        .selectAll('g')
        .data(links)
        .enter()
        .append('g')
        .attr('class', 'link');
      
      // 连线本体
      link.append('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', d => Math.sqrt(d.value) * 2);
      
      // 连线标签
      link.append('text')
        .attr('class', 'link-label')
        .attr('dy', -5)
        .attr('text-anchor', 'middle')
        .text(d => d.label);
      
      // 绘制节点
      const node = mainGroup.append('g')
        .attr('class', 'nodes')
        .selectAll('.node')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .on('click', (event, d) => {
          event.stopPropagation();
          selectedNode.value = d;
          emit('nodeClick', d);
        })
        .call(d3.drag()
          .on('start', dragstarted)
          .on('drag', dragged)
          .on('end', dragended));
      
      // 节点圆
      node.append('circle')
        .attr('r', d => {
          if (centerNode.value && d.id === centerNode.value.id) {
            return parseInt(nodeSize.value) * 1.3;
          }
          return nodeSize.value;
        })
        .attr('fill', d => nodeColors.value[d.type] || '#999')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);
      
      // 节点标签
      node.append('text')
        .attr('dy', 4)
        .attr('text-anchor', 'middle')
        .text(d => d.name)
        .attr('font-size', 12)
        .attr('fill', '#fff');
      
      // 更新位置
      simulation.value.on('tick', () => {
        link.select('line')
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);
        
        link.select('text')
          .attr('x', d => (d.source.x + d.target.x) / 2)
          .attr('y', d => (d.source.y + d.target.y) / 2);
        
        node.attr('transform', d => `translate(${d.x},${d.y})`);
      });
      
      // 拖拽函数
      function dragstarted(event, d) {
        if (!event.active) simulation.value.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }
      
      function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
      }
      
      function dragended(event, d) {
        if (!event.active) simulation.value.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }
      
      // 点击空白区域取消选择
      svg.value.on('click', () => {
        selectedNode.value = null;
      });
    };
    
    // 监听数据变化
    watch(() => props.graphData, (newVal, oldVal) => {
      if (newVal && newVal !== oldVal) {
        initGraph();
      }
    }, { deep: true });
    
    // 生命周期钩子
    onMounted(() => {
      initGraph();
    });
    
    onUnmounted(() => {
      // 清理
      if (simulation.value) {
        simulation.value.stop();
      }
    });
    
    return {
      graphContainer,
      nodeSize,
      linkStrength,
      nodeColors,
      showControls,
      showLegend,
      selectedNode,
      centerNode,
      activeNodeTypes,
      selectedNodeRelations,
      toggleControls,
      updateGraph,
      resetGraph,
      exportGraph,
      filterNodes,
      formatNodeType,
      formatPropertyName,
      selectNodeById
    };
  }
};
</script>

<style scoped>
.knowledge-graph-container {
  position: relative;
  width: 100%;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background-color: #fafafa;
  overflow: hidden;
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

.graph-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.8rem 1rem;
  border-bottom: 1px solid #e0e0e0;
  background-color: #f5f5f5;
}

.graph-header h3 {
  margin: 0;
  color: #333;
  font-size: 1.1rem;
  font-weight: 500;
}

.graph-actions {
  display: flex;
  gap: 0.5rem;
}

.action-button {
  padding: 0.4rem 0.8rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: white;
  color: #0066cc;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s;
}

.action-button:hover {
  background-color: #f0f7ff;
  border-color: #0066cc;
}

.graph-viewport {
  position: relative;
  width: 100%;
  height: 600px;
}

.graph-container {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.graph-controls {
  position: absolute;
  top: 0;
  right: 0;
  padding: 1rem;
  background-color: rgba(255, 255, 255, 0.9);
  border-left: 1px solid #eee;
  border-bottom: 1px solid #eee;
  border-bottom-left-radius: 8px;
  z-index: 10;
  width: 250px;
  max-height: 80%;
  overflow-y: auto;
  box-shadow: -2px 2px 8px rgba(0, 0, 0, 0.1);
}

.control-group {
  margin-bottom: 1rem;
}

.control-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
  color: #555;
}

.filter-section {
  margin-top: 1.5rem;
}

.filter-section h4 {
  margin: 0 0 0.8rem 0;
  font-size: 0.95rem;
  color: #444;
}

.node-type-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.node-type-option {
  display: flex;
  align-items: center;
  padding: 0.3rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 0.85rem;
  cursor: pointer;
  margin-right: 0.4rem;
  margin-bottom: 0.4rem;
}

.node-type-option input {
  margin-right: 0.3rem;
}

.graph-legend {
  position: absolute;
  bottom: 1rem;
  left: 1rem;
  padding: 0.8rem;
  background-color: rgba(255, 255, 255, 0.85);
  border-radius: 6px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  max-width: 180px;
}

.graph-legend h4 {
  margin: 0 0 0.5rem 0;
  font-size: 0.9rem;
  color: #444;
}

.legend-item {
  display: flex;
  align-items: center;
  margin-bottom: 0.4rem;
}

.legend-color {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 6px;
}

.legend-label {
  font-size: 0.85rem;
  color: #333;
}

.node-details {
  position: absolute;
  top: 1rem;
  left: 1rem;
  width: 300px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.15);
  z-index: 10;
}

.node-header {
  display: flex;
  align-items: center;
  padding: 0.8rem 1rem;
  border-bottom: 1px solid #eee;
  position: relative;
}

.node-header h4 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 500;
  color: #333;
  flex: 1;
}

.node-type-badge {
  padding: 0.3rem 0.6rem;
  border-radius: 12px;
  color: white;
  font-size: 0.75rem;
  margin-left: 0.5rem;
}

.close-button {
  background: none;
  border: none;
  color: #999;
  font-size: 1.2rem;
  cursor: pointer;
  padding: 0.2rem;
  margin-left: 0.5rem;
}

.close-button:hover {
  color: #666;
}

.node-properties {
  padding: 1rem;
  border-bottom: 1px solid #eee;
}

.property-item {
  margin-bottom: 0.6rem;
  line-height: 1.4;
}

.property-item strong {
  font-weight: 500;
  color: #555;
}

.node-relations {
  padding: 1rem;
}

.node-relations h5 {
  margin: 0 0 0.8rem 0;
  color: #444;
  font-size: 0.95rem;
  font-weight: 500;
}

.relation-item {
  padding: 0.6rem;
  margin-bottom: 0.5rem;
  background-color: #f9f9f9;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.relation-item:hover {
  background-color: #f0f7ff;
}

.relation-name {
  font-size: 0.85rem;
  color: #666;
  display: block;
  margin-bottom: 0.3rem;
}

.target-node {
  font-weight: 500;
  color: #0066cc;
  margin-right: 0.5rem;
}

.target-node-type {
  display: inline-block;
  padding: 0.2rem 0.4rem;
  border-radius: 10px;
  color: white;
  font-size: 0.7rem;
}

.info-panel {
  padding: 0.5rem 1rem;
  background-color: #f5f5f5;
  font-size: 0.85rem;
  color: #666;
  border-top: 1px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
}

/* D3 样式 */
:deep(.link line) {
  stroke-opacity: 0.6;
  transition: stroke-width 0.2s;
}

:deep(.link:hover line) {
  stroke-width: 3;
  stroke-opacity: 0.8;
}

:deep(.link-label) {
  fill: #555;
  font-size: 10px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
}

:deep(.link:hover .link-label) {
  opacity: 1;
}

:deep(.node circle) {
  transition: r 0.3s, fill 0.3s;
}

:deep(.node:hover circle) {
  fill-opacity: 0.8;
  stroke-width: 3;
}

:deep(.node text) {
  pointer-events: none;
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
  font-size: 12px;
  text-shadow: 0 1px 1px rgba(0, 0, 0, 0.3);
}
</style> 