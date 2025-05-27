import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver
import json

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

@dataclass
class Entity:
    """实体类"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]

@dataclass
class Relation:
    """关系类"""
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any]

@dataclass
class GraphQuery:
    """图查询结果"""
    entities: List[Entity]
    relations: List[Relation]
    paths: List[List[str]]

class KnowledgeGraphConfig:
    """知识图谱配置"""
    
    def __init__(self):
        self.neo4j_uri = getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = getattr(settings, 'NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = getattr(settings, 'NEO4J_PASSWORD', 'password')
        self.batch_size = 1000

class KnowledgeGraph:
    """知识图谱主类"""
    
    def __init__(self, config: KnowledgeGraphConfig = None):
        self.config = config or KnowledgeGraphConfig()
        self.driver: Optional[AsyncDriver] = None
        
    async def initialize(self):
        """初始化Neo4j连接"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_username, self.config.neo4j_password)
            )
            
            # 测试连接
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                await result.consume()
            
            logger.info("知识图谱连接初始化成功")
            
        except Exception as e:
            logger.error(f"知识图谱连接初始化失败: {e}")
            raise
    
    async def close(self):
        """关闭连接"""
        if self.driver:
            await self.driver.close()
    
    async def create_entity(self, entity: Entity) -> bool:
        """创建实体节点"""
        try:
            query = f"""
            MERGE (e:{entity.type} {{id: $id}})
            SET e.name = $name
            SET e += $properties
            RETURN e
            """
            
            async with self.driver.session() as session:
                await session.run(query, {
                    'id': entity.id,
                    'name': entity.name,
                    'properties': entity.properties
                })
            
            logger.debug(f"创建实体成功: {entity.name}")
            return True
            
        except Exception as e:
            logger.error(f"创建实体失败: {e}")
            return False
    
    async def create_relation(self, relation: Relation) -> bool:
        """创建关系"""
        try:
            query = f"""
            MATCH (a {{id: $source}})
            MATCH (b {{id: $target}})
            MERGE (a)-[r:{relation.type}]->(b)
            SET r += $properties
            RETURN r
            """
            
            async with self.driver.session() as session:
                await session.run(query, {
                    'source': relation.source,
                    'target': relation.target,
                    'properties': relation.properties
                })
            
            logger.debug(f"创建关系成功: {relation.type}")
            return True
            
        except Exception as e:
            logger.error(f"创建关系失败: {e}")
            return False
    
    async def query_entity_neighbors(self, entity_id: str, max_depth: int = 2) -> GraphQuery:
        """查询实体的邻居节点"""
        try:
            query = f"""
            MATCH (e {{id: $entity_id}})
            CALL apoc.path.subgraphNodes(e, {{
                maxLevel: $max_depth,
                relationshipFilter: '>',
                labelFilter: '+'
            }}) YIELD node
            OPTIONAL MATCH (node)-[r]-(connected)
            WHERE connected IN apoc.path.subgraphNodes(e, {{maxLevel: $max_depth}})
            RETURN DISTINCT node, r, connected
            """
            
            entities = []
            relations = []
            
            async with self.driver.session() as session:
                result = await session.run(query, {
                    'entity_id': entity_id,
                    'max_depth': max_depth
                })
                
                async for record in result:
                    node = record.get('node')
                    if node:
                        entity = Entity(
                            id=node.get('id'),
                            name=node.get('name', ''),
                            type=list(node.labels)[0] if node.labels else 'Unknown',
                            properties=dict(node)
                        )
                        entities.append(entity)
                    
                    rel = record.get('r')
                    if rel:
                        relation = Relation(
                            id=str(rel.id),
                            source=str(rel.start_node.id),
                            target=str(rel.end_node.id),
                            type=rel.type,
                            properties=dict(rel)
                        )
                        relations.append(relation)
            
            return GraphQuery(entities=entities, relations=relations, paths=[])
            
        except Exception as e:
            logger.error(f"查询实体邻居失败: {e}")
            return GraphQuery(entities=[], relations=[], paths=[])
    
    async def query_path(self, start_id: str, end_id: str, max_length: int = 5) -> List[List[str]]:
        """查询两个实体之间的路径"""
        try:
            query = f"""
            MATCH (start {{id: $start_id}}), (end {{id: $end_id}})
            MATCH path = shortestPath((start)-[*1..{max_length}]-(end))
            RETURN [node in nodes(path) | node.id] as path
            LIMIT 10
            """
            
            paths = []
            async with self.driver.session() as session:
                result = await session.run(query, {
                    'start_id': start_id,
                    'end_id': end_id
                })
                
                async for record in result:
                    path = record.get('path', [])
                    if path:
                        paths.append(path)
            
            return paths
            
        except Exception as e:
            logger.error(f"查询路径失败: {e}")
            return []
    
    async def search_entities_by_name(self, name: str, entity_type: str = None) -> List[Entity]:
        """根据名称搜索实体"""
        try:
            type_filter = f":{entity_type}" if entity_type else ""
            query = f"""
            MATCH (e{type_filter})
            WHERE e.name CONTAINS $name OR e.id CONTAINS $name
            RETURN e
            LIMIT 50
            """
            
            entities = []
            async with self.driver.session() as session:
                result = await session.run(query, {'name': name})
                
                async for record in result:
                    node = record.get('e')
                    if node:
                        entity = Entity(
                            id=node.get('id'),
                            name=node.get('name', ''),
                            type=list(node.labels)[0] if node.labels else 'Unknown',
                            properties=dict(node)
                        )
                        entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"搜索实体失败: {e}")
            return []

class DiseaseKnowledgeGraph(KnowledgeGraph):
    """水生动物疾病知识图谱"""
    
    def __init__(self, config: KnowledgeGraphConfig = None):
        super().__init__(config)
        
    async def create_disease_schema(self):
        """创建疾病知识图谱模式"""
        try:
            # 创建索引
            indexes = [
                "CREATE INDEX disease_id IF NOT EXISTS FOR (d:Disease) ON (d.id)",
                "CREATE INDEX animal_id IF NOT EXISTS FOR (a:Animal) ON (a.id)",
                "CREATE INDEX symptom_id IF NOT EXISTS FOR (s:Symptom) ON (s.id)",
                "CREATE INDEX treatment_id IF NOT EXISTS FOR (t:Treatment) ON (t.id)",
                "CREATE INDEX pathogen_id IF NOT EXISTS FOR (p:Pathogen) ON (p.id)",
            ]
            
            async with self.driver.session() as session:
                for index_query in indexes:
                    await session.run(index_query)
            
            logger.info("疾病知识图谱模式创建成功")
            
        except Exception as e:
            logger.error(f"创建疾病知识图谱模式失败: {e}")
    
    async def add_disease_data(self, disease_data: Dict[str, Any]) -> bool:
        """添加疾病数据"""
        try:
            # 创建疾病节点
            disease = Entity(
                id=disease_data['id'],
                name=disease_data['name'],
                type='Disease',
                properties={
                    'description': disease_data.get('description', ''),
                    'severity': disease_data.get('severity', 'unknown'),
                    'category': disease_data.get('category', 'unknown')
                }
            )
            await self.create_entity(disease)
            
            # 创建症状关系
            for symptom in disease_data.get('symptoms', []):
                symptom_entity = Entity(
                    id=f"symptom_{symptom['id']}",
                    name=symptom['name'],
                    type='Symptom',
                    properties=symptom.get('properties', {})
                )
                await self.create_entity(symptom_entity)
                
                relation = Relation(
                    id=f"has_symptom_{disease.id}_{symptom_entity.id}",
                    source=disease.id,
                    target=symptom_entity.id,
                    type='HAS_SYMPTOM',
                    properties={'frequency': symptom.get('frequency', 'unknown')}
                )
                await self.create_relation(relation)
            
            # 创建治疗关系
            for treatment in disease_data.get('treatments', []):
                treatment_entity = Entity(
                    id=f"treatment_{treatment['id']}",
                    name=treatment['name'],
                    type='Treatment',
                    properties=treatment.get('properties', {})
                )
                await self.create_entity(treatment_entity)
                
                relation = Relation(
                    id=f"treated_by_{disease.id}_{treatment_entity.id}",
                    source=disease.id,
                    target=treatment_entity.id,
                    type='TREATED_BY',
                    properties={'effectiveness': treatment.get('effectiveness', 'unknown')}
                )
                await self.create_relation(relation)
            
            return True
            
        except Exception as e:
            logger.error(f"添加疾病数据失败: {e}")
            return False
    
    async def diagnose_by_symptoms(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """基于症状诊断疾病"""
        try:
            symptom_conditions = " OR ".join([f"s.name CONTAINS '{symptom}'" for symptom in symptoms])
            
            query = f"""
            MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
            WHERE {symptom_conditions}
            WITH d, count(s) as matched_symptoms
            OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(all_symptoms:Symptom)
            WITH d, matched_symptoms, count(all_symptoms) as total_symptoms
            RETURN d.id as disease_id, d.name as disease_name, d.description as description,
                   matched_symptoms, total_symptoms,
                   toFloat(matched_symptoms) / toFloat(total_symptoms) as confidence
            ORDER BY confidence DESC, matched_symptoms DESC
            LIMIT 10
            """
            
            results = []
            async with self.driver.session() as session:
                result = await session.run(query)
                
                async for record in result:
                    results.append({
                        'disease_id': record.get('disease_id'),
                        'disease_name': record.get('disease_name'),
                        'description': record.get('description'),
                        'matched_symptoms': record.get('matched_symptoms'),
                        'total_symptoms': record.get('total_symptoms'),
                        'confidence': record.get('confidence', 0.0)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"症状诊断失败: {e}")
            return [] 