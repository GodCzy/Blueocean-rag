import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Grid,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
  Paper
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Medical as MedicalIcon,
  Assessment as AssessmentIcon,
  Water as WaterIcon,
  BugReport as BugIcon
} from '@mui/icons-material';
import { diagnosisAPI } from '../services/api';

const DiagnosisPage = () => {
  const [formData, setFormData] = useState({
    animal_type: '',
    symptoms: [],
    environment_info: {},
    water_parameters: {},
    use_knowledge_graph: true,
    use_rag: true
  });

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  
  const [availableSymptoms, setAvailableSymptoms] = useState([]);
  const [availableAnimals, setAvailableAnimals] = useState([]);
  const [selectedSymptom, setSelectedSymptom] = useState('');

  // 获取可用的症状和动物类型
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [symptomsRes, animalsRes] = await Promise.all([
          diagnosisAPI.getSymptoms(),
          diagnosisAPI.getAnimalTypes()
        ]);
        setAvailableSymptoms(symptomsRes.symptoms);
        setAvailableAnimals(animalsRes.animal_types);
      } catch (err) {
        console.error('获取基础数据失败:', err);
      }
    };
    fetchData();
  }, []);

  const handleAnimalTypeChange = (event) => {
    setFormData(prev => ({
      ...prev,
      animal_type: event.target.value
    }));
  };

  const handleAddSymptom = () => {
    if (selectedSymptom && !formData.symptoms.includes(selectedSymptom)) {
      setFormData(prev => ({
        ...prev,
        symptoms: [...prev.symptoms, selectedSymptom]
      }));
      setSelectedSymptom('');
    }
  };

  const handleRemoveSymptom = (symptom) => {
    setFormData(prev => ({
      ...prev,
      symptoms: prev.symptoms.filter(s => s !== symptom)
    }));
  };

  const handleWaterParameterChange = (key, value) => {
    setFormData(prev => ({
      ...prev,
      water_parameters: {
        ...prev.water_parameters,
        [key]: value
      }
    }));
  };

  const handleEnvironmentChange = (key, value) => {
    setFormData(prev => ({
      ...prev,
      environment_info: {
        ...prev.environment_info,
        [key]: value
      }
    }));
  };

  const handleSubmit = async () => {
    if (!formData.animal_type || formData.symptoms.length === 0) {
      setError('请选择动物类型并至少添加一个症状');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await diagnosisAPI.diagnose(formData);
      setResults(response);
    } catch (err) {
      setError(err.message || '诊断失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  const getDiseaseConfidenceColor = (confidence) => {
    if (confidence > 0.8) return 'success';
    if (confidence > 0.6) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <MedicalIcon color="primary" />
        蓝海智询 - 水生动物疾病诊断
      </Typography>

      <Grid container spacing={3}>
        {/* 输入表单 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                诊断信息输入
              </Typography>

              {/* 动物类型选择 */}
              <FormControl fullWidth margin="normal">
                <InputLabel>动物类型</InputLabel>
                <Select
                  value={formData.animal_type}
                  onChange={handleAnimalTypeChange}
                  label="动物类型"
                >
                  {availableAnimals.map((animal) => (
                    <MenuItem key={animal.name} value={animal.name}>
                      {animal.name} ({animal.category})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* 症状选择 */}
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  症状选择
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                  <FormControl sx={{ minWidth: 200 }}>
                    <InputLabel>选择症状</InputLabel>
                    <Select
                      value={selectedSymptom}
                      onChange={(e) => setSelectedSymptom(e.target.value)}
                      label="选择症状"
                    >
                      {availableSymptoms.map((symptom) => (
                        <MenuItem key={symptom.name} value={symptom.name}>
                          {symptom.name} ({symptom.category})
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <Button
                    variant="contained"
                    onClick={handleAddSymptom}
                    disabled={!selectedSymptom}
                  >
                    添加
                  </Button>
                </Box>

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {formData.symptoms.map((symptom) => (
                    <Chip
                      key={symptom}
                      label={symptom}
                      onDelete={() => handleRemoveSymptom(symptom)}
                      color="primary"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Box>

              {/* 环境信息 */}
              <Accordion sx={{ mt: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <WaterIcon />
                    <Typography>水质参数 (可选)</Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="温度 (°C)"
                        type="number"
                        onChange={(e) => handleWaterParameterChange('temperature', e.target.value)}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="pH值"
                        type="number"
                        inputProps={{ step: 0.1 }}
                        onChange={(e) => handleWaterParameterChange('pH', e.target.value)}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="溶解氧 (mg/L)"
                        type="number"
                        inputProps={{ step: 0.1 }}
                        onChange={(e) => handleWaterParameterChange('dissolved_oxygen', e.target.value)}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        fullWidth
                        label="盐度 (‰)"
                        type="number"
                        onChange={(e) => handleWaterParameterChange('salinity', e.target.value)}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* 其他环境信息 */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>其他环境信息 (可选)</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="养殖密度"
                        onChange={(e) => handleEnvironmentChange('density', e.target.value)}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="饲料类型"
                        onChange={(e) => handleEnvironmentChange('feed_type', e.target.value)}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="养殖天数"
                        type="number"
                        onChange={(e) => handleEnvironmentChange('culture_days', e.target.value)}
                      />
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}

              <Button
                variant="contained"
                fullWidth
                size="large"
                onClick={handleSubmit}
                disabled={loading || !formData.animal_type || formData.symptoms.length === 0}
                sx={{ mt: 3 }}
              >
                {loading ? <CircularProgress size={24} /> : '开始诊断'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* 诊断结果 */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AssessmentIcon />
                诊断结果
              </Typography>

              {!results && !loading && (
                <Paper sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
                  <BugIcon sx={{ fontSize: 48, mb: 1 }} />
                  <Typography>
                    请填写诊断信息并点击"开始诊断"
                  </Typography>
                </Paper>
              )}

              {loading && (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              )}

              {results && (
                <Box>
                  {/* 疾病候选 */}
                  <Accordion defaultExpanded>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">可能的疾病</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List>
                        {results.disease_candidates?.map((disease, index) => (
                          <React.Fragment key={index}>
                            <ListItem>
                              <ListItemText
                                primary={
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Typography variant="subtitle2">
                                      {disease.disease_name}
                                    </Typography>
                                    <Chip
                                      size="small"
                                      label={`${Math.round((results.confidence_scores?.[index] || 0) * 100)}%`}
                                      color={getDiseaseConfidenceColor(results.confidence_scores?.[index] || 0)}
                                    />
                                  </Box>
                                }
                                secondary={disease.description}
                              />
                            </ListItem>
                            {index < results.disease_candidates.length - 1 && <Divider />}
                          </React.Fragment>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>

                  {/* 治疗建议 */}
                  {results.treatment_recommendations && (
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="subtitle1">治疗建议</Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        {results.treatment_recommendations.map((treatment, index) => (
                          <Paper key={index} sx={{ p: 2, mb: 1 }}>
                            <Typography variant="subtitle2" gutterBottom>
                              针对: {treatment.for_disease}
                            </Typography>
                            <Typography variant="body2">
                              {treatment.treatment}
                            </Typography>
                          </Paper>
                        ))}
                      </AccordionDetails>
                    </Accordion>
                  )}

                  {/* 环境分析 */}
                  {results.environment_analysis && (
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="subtitle1">环境分析</Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Typography variant="body2">
                          {results.environment_analysis}
                        </Typography>
                      </AccordionDetails>
                    </Accordion>
                  )}

                  {/* RAG回答 */}
                  {results.rag_response && (
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="subtitle1">知识库检索结果</Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Typography variant="body2">
                          {results.rag_response}
                        </Typography>
                      </AccordionDetails>
                    </Accordion>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DiagnosisPage; 