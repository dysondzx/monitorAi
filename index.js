const { createApp, ref, onMounted, computed } = Vue;
        
createApp({
    setup() {
        // 状态和数据
        const deviceStatus = ref('normal');
        const trafficStatus = ref('normal');
        const modelReady = ref(false);
        const isMonitoring = ref(false);
        const logs = ref([]);
        const memoryUsage = ref(0);
        const accuracy = ref(92);
        const anomalyDetectionRate = ref(88);
        const processedDataPoints = ref(0);
        const modelLoadProgress = ref(0);
        const selectedSpeed = ref(2000);
        
        const speeds = ref([
            { label: '慢速', value: 3000 },
            { label: '中速', value: 2000 },
            { label: '快速', value: 1000 }
        ]);
        
        // 模型引用
        let trafficModel = null;
        let imageModel = null;
        
        // DOM 引用
        let deviceCtx = null;
        let trafficCtx = null;
        
        // 计算属性
        const deviceStatusDisplay = computed(() => {
            return {
                'normal': '正常运作',
                'warning': '轻微异常',
                'danger': '严重故障'
            }[deviceStatus.value];
        });
        
        const trafficStatusDisplay = computed(() => {
            return {
                'normal': '流量正常',
                'warning': '流量偏高',
                'danger': '网络风暴'
            }[trafficStatus.value];
        });
        
        // 日志记录方法
        const addLog = (message, type = 'info') => {
            const timestamp = new Date().toLocaleTimeString();
            logs.value.unshift({
                message: `[${timestamp}] ${message}`,
                type
            });
            
            if (logs.value.length > 100) {
                logs.value.pop();
            }
        };
        
        // TensorFlow.js 模型初始化
        const initModels = async () => {
            addLog('开始加载机器学习模型...');
            modelLoadProgress.value = 0;
            
            try {
                // 1. 加载流量异常检测模型
                trafficModel = await createTrafficModel();
                modelLoadProgress.value = 50;
                
                // 2. 加载图像识别模型
                imageModel = await createImageModel();
                modelLoadProgress.value = 100;
                
                modelReady.value = true;
                addLog('所有模型加载完成，系统准备就绪!', 'info');
            } catch (error) {
                addLog(`模型加载失败: ${error.message}`, 'danger');
                modelLoadProgress.value = 0;
            }
        };
        
        // 创建流量异常检测模型（纯前端实现）
        const createTrafficModel = () => {
            return new Promise((resolve) => {
                addLog('创建流量异常检测模型...');
                
                // 创建简单的时序预测模型
                const model = tf.sequential();
                model.add(tf.layers.lstm({
                    units: 32,
                    inputShape: [10, 1],
                    returnSequences: false
                }));
                model.add(tf.layers.dense({ units: 1 }));
                
                // 编译模型
                model.compile({
                    optimizer: tf.train.adam(),
                    loss: 'meanSquaredError'
                });
                
                // 模拟训练数据并简单训练
                const trainModel = async () => {
                    const data = generateTrainingData();
                    const xs = tf.tensor3d(data.inputs);
                    const ys = tf.tensor2d(data.labels);
                    
                    await model.fit(xs, ys, {
                        epochs: 10,
                        batchSize: 16,
                        callbacks: {
                            onEpochEnd: (epoch, logs) => {
                                const current = (epoch + 1) * 10;
                                modelLoadProgress.value = current > 50 ? 50 : current;
                                addLog(`流量模型训练中 (epoch ${epoch + 1}) - Loss: ${logs.loss.toFixed(4)}`);
                            }
                        }
                    });
                    
                    tf.dispose([xs, ys]);
                    resolve(model);
                };
                
                trainModel();
            });
        };
        
        // 创建图像识别模型（基于迁移学习）
        const createImageModel = () => {
            return new Promise((resolve) => {
                addLog('创建设备状态识别模型...');
                
                // 使用预训练的MobileNet作为基础模型
                const loadModel = async () => {
                    const baseModel = await tf.loadLayersModel(
                        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
                    );
                    
                    // 冻结基础模型的权重
                    baseModel.trainable = false;
                    
                    // 创建迁移学习模型
                    const model = tf.sequential();
                    model.add(tf.layers.inputLayer({ inputShape: [224, 224, 3] }));
                    model.add(tf.layers.resizing({ height: 224, width: 224 }));
                    model.add(baseModel);
                    model.add(tf.layers.globalAveragePooling2d());
                    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
                    
                    // 编译模型
                    model.compile({
                        optimizer: tf.train.adam(0.001),
                        loss: 'categoricalCrossentropy',
                        metrics: ['accuracy']
                    });
                    
                    // 模拟训练
                    const mockTrain = async () => {
                        // 由于是演示，跳过真实训练
                        const progressInterval = setInterval(() => {
                            if (modelLoadProgress.value < 100) {
                                modelLoadProgress.value += 5;
                            } else {
                                clearInterval(progressInterval);
                            }
                        }, 100);
                        
                        addLog('图像模型训练中...');
                        await new Promise(resolve => setTimeout(resolve, 1500));
                        resolve(model);
                    };
                    
                    mockTrain();
                };
                
                loadModel().then(resolve);
            });
        };
        
        // 生成训练数据（纯前端模拟）
        const generateTrainingData = () => {
            const inputs = [];
            const labels = [];
            
            for (let i = 0; i < 1000; i++) {
                const seq = Array.from({ length: 10 }, () => Math.random() * 100);
                inputs.push(seq.map(v => [v]));
                
                // 简单规则：如果最后3个值中有1个大于80则视为异常
                const lastThree = seq.slice(-3);
                const isAnomaly = lastThree.some(v => v > 80) ? 1 : 0;
                labels.push([isAnomaly]);
            }
            
            return { inputs, labels };
        };
        
        // 生成设备模拟图像
        const drawDeviceImage = (status) => {
            if (!deviceCtx) return;
            
            const ctx = deviceCtx;
            const width = ctx.canvas.width;
            const height = ctx.canvas.height;
            
            // 清除画布
            ctx.clearRect(0, 0, width, height);
            
            // 根据状态设置背景颜色
            let bgColor = '#e8f5e9';
            if (status === 'warning') bgColor = '#fff8e1';
            if (status === 'danger') bgColor = '#ffebee';
            
            ctx.fillStyle = bgColor;
            ctx.fillRect(0, 0, width, height);
            
            // 绘制设备主体
            ctx.fillStyle = '#546e7a';
            ctx.fillRect(width * 0.2, height * 0.2, width * 0.6, height * 0.6);
            
            // 绘制状态指示灯
            ctx.fillStyle = status === 'normal' ? '#4caf50' : 
                            status === 'warning' ? '#ff9800' : '#f44336';
            ctx.beginPath();
            ctx.arc(width * 0.8, height * 0.3, width * 0.04, 0, Math.PI * 2);
            ctx.fill();
            
            // 添加状态文本
            ctx.fillStyle = '#37474f';
            ctx.font = 'bold 16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('工业控制设备', width * 0.5, height * 0.8);
            
            // 状态信息
            ctx.font = '20px Arial';
            ctx.fillText(deviceStatusDisplay.value, width * 0.5, height * 0.9);
            
            // 模拟缺陷 - 基于状态
            if (status !== 'normal') {
                ctx.strokeStyle = '#d32f2f';
                ctx.lineWidth = 3;
                
                if (status === 'warning') {
                    // 轻微异常 - 闪烁效果
                    if (Date.now() % 600 < 300) {
                        ctx.beginPath();
                        ctx.moveTo(width * 0.4, height * 0.35);
                        ctx.lineTo(width * 0.6, height * 0.5);
                        ctx.stroke();
                    }
                } else {
                    // 严重故障 - 十字标记
                    ctx.beginPath();
                    ctx.moveTo(width * 0.35, height * 0.35);
                    ctx.lineTo(width * 0.65, height * 0.65);
                    ctx.moveTo(width * 0.65, height * 0.35);
                    ctx.lineTo(width * 0.35, height * 0.65);
                    ctx.stroke();
                }
            }
        };
        
        // 绘制流量图表
        const drawTrafficChart = (data, status) => {
            if (!trafficCtx) return;
            
            const ctx = trafficCtx;
            const width = ctx.canvas.width;
            const height = ctx.canvas.height;
            
            // 清除画布
            ctx.clearRect(0, 0, width, height);
            
            // 背景
            const bgColor = status === 'normal' ? '#e3f2fd' : 
                            status === 'warning' ? '#fffde7' : '#ffebee';
            ctx.fillStyle = bgColor;
            ctx.fillRect(0, 0, width, height);
            
            // 网格线
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 0.5;
            
            // 水平网格线
            const gridLines = 5;
            for (let i = 1; i < gridLines; i++) {
                ctx.beginPath();
                ctx.moveTo(0, i * height / gridLines);
                ctx.lineTo(width, i * height / gridLines);
                ctx.stroke();
            }
            
            // 图表边框
            ctx.strokeStyle = '#bdbdbd';
            ctx.lineWidth = 1;
            ctx.strokeRect(0, 0, width, height);
            
            // 流量阈值线
            ctx.strokeStyle = status === 'normal' ? '#388e3c' : 
                            status === 'warning' ? '#f57c00' : '#d32f2f';
            ctx.lineWidth = 2;
            
            // 根据状态绘制不同的阈值线
            const warningThreshold = 70;
            const dangerThreshold = 85;
            
            // 警告线
            const yWarning = height * (1 - warningThreshold / 100);
            ctx.beginPath();
            ctx.setLineDash([5, 3]);
            ctx.moveTo(0, yWarning);
            ctx.lineTo(width, yWarning);
            ctx.stroke();
            
            // 危险线
            const yDanger = height * (1 - dangerThreshold / 100);
            ctx.beginPath();
            ctx.setLineDash([5, 3]);
            ctx.moveTo(0, yDanger);
            ctx.lineTo(width, yDanger);
            ctx.stroke();
            ctx.setLineDash([]);
            
            // 绘制数据点
            ctx.strokeStyle = '#1976d2';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            const pointWidth = width / (data.length - 1);
            
            for (let i = 0; i < data.length; i++) {
                const x = i * pointWidth;
                const y = height * (1 - data[i] / 100);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                
                // 标记异常点
                if (data[i] > warningThreshold) {
                    ctx.fillStyle = data[i] > dangerThreshold ? '#d32f2f' : '#f57c00';
                    ctx.beginPath();
                    ctx.arc(x, y, 3, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            
            ctx.stroke();
            
            // 添加标题
            ctx.fillStyle = '#37474f';
            ctx.font = 'bold 16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('网络流量监控', width * 0.5, 20);
            ctx.fillText(trafficStatusDisplay.value, width * 0.5, height - 10);
        };
        
        // 生成模拟流量数据
        const generateTrafficData = () => {
            const base = 30 + Math.random() * 15; // 30-45的正常范围
            let data = [];
            
            // 生成10个点
            for (let i = 0; i < 10; i++) {
                // 正常波动
                let value = base + Math.random() * 10 - 5;
                
                // 10%几率触发流量异常
                if (Math.random() < 0.1) {
                    // 轻度或重度异常
                    const severity = Math.random() < 0.5 ? 25 + Math.random() * 20 : 50 + Math.random() * 40;
                    value += severity;
                    
                    // 确保不超过100%
                    if (value > 100) value = 100;
                }
                
                data.push(value);
            }
            
            return data;
        };
        
        // 使用模型进行流量异常检测
        const detectTrafficAnomaly = async (data) => {
            if (!trafficModel || !modelReady.value) return;
            
            try {
                // 将数据转换为Tensor
                const input = tf.tensor3d([data.map(v => [v / 100])]);
                
                // 进行预测
                const prediction = await trafficModel.predict(input);
                const value = prediction.dataSync()[0];
                
                // 清理内存
                tf.dispose([input, prediction]);
                
                // 根据预测值确定状态
                if (value > 0.6) {
                    trafficStatus.value = 'danger';
                    return 'danger';
                } else if (value > 0.4) {
                    trafficStatus.value = 'warning';
                    return 'warning';
                } else {
                    trafficStatus.value = 'normal';
                    return 'normal';
                }
            } catch (error) {
                addLog(`流量检测失败: ${error.message}`, 'warning');
                return 'normal';
            }
        };
        
        // 图像识别处理流程
        const processDeviceImage = async () => {
            if (!imageModel || !deviceCtx || !modelReady.value) return;
            
            try {
                const canvas = deviceCtx.canvas;
                
                // 从Canvas获取图像数据
                const imageData = deviceCtx.getImageData(0, 0, canvas.width, canvas.height);
                
                // 创建Tensor
                let tensor = tf.browser.fromPixels(imageData);
                
                // 调整尺寸以匹配模型输入
                tensor = tf.image.resizeBilinear(tensor, [224, 224]);
                
                // 归一化
                tensor = tensor.div(255.0);
                
                // 增加批次维度
                tensor = tensor.expandDims(0);
                
                // 进行预测
                const prediction = await imageModel.predict(tensor);
                const values = prediction.dataSync();
                
                // 确定设备状态 (正常/警告/危险)
                const predictedStatusIndex = values.indexOf(Math.max(...values));
                const status = ['normal', 'warning', 'danger'][predictedStatusIndex];
                
                // 清理内存
                tf.dispose([tensor, prediction]);
                
                // 更新状态
                deviceStatus.value = status;
                
                return status;
            } catch (error) {
                addLog(`图像识别失败: ${error.message}`, 'warning');
                return 'normal';
            }
        };
        
        // 监控任务
        const monitoringTask = async () => {
            if (!isMonitoring.value) return;
            
            processedDataPoints.value++;
            
            // 任务1: 设备状态识别
            drawDeviceImage(deviceStatus.value);
            const imgStatus = await processDeviceImage();
            
            // 任务2: 流量异常检测
            const trafficData = generateTrafficData();
            const trafficStatus = await detectTrafficAnomaly(trafficData);
            drawTrafficChart(trafficData, trafficStatus);
            
            // 模拟内存使用变化
            memoryUsage.value = (Math.random() * 2 + 8).toFixed(1);
            
            // 每50个数据点更新一次准确率
            if (processedDataPoints.value % 50 === 0) {
                accuracy.value = Math.max(85, Math.min(99, accuracy.value + (Math.random() * 2 - 1)));
                anomalyDetectionRate.value = Math.max(85, Math.min(97, anomalyDetectionRate.value + (Math.random() * 2 - 1)));
            }
            
            // 随机生成日志信息
            if (Math.random() < 0.2) {
                const normalMsgs = [
                    '系统运行正常，所有组件工作稳定',
                    '网络流量波动在正常范围内',
                    '设备温度稳定在安全区间'
                ];
                
                const warningMsgs = [
                    '流量高峰检测 - 持续监控中',
                    '设备CPU使用率偏高',
                    '检测到轻微网络延迟'
                ];
                
                const dangerMsgs = [
                    '警告! 检测到网络风暴!',
                    '设备温度超限，请立即检查!',
                    '严重: 网络连接中断'
                ];
                
                const msgs = trafficStatus === 'normal' ? normalMsgs : 
                            trafficStatus === 'warning' ? warningMsgs : dangerMsgs;
                
                const msgType = trafficStatus === 'normal' ? 'info' : 
                                trafficStatus === 'warning' ? 'warning' : 'danger';
                
                const randomMsg = msgs[Math.floor(Math.random() * msgs.length)];
                addLog(randomMsg, msgType);
            }
            
            // 下一个循环
            if (isMonitoring.value) {
                setTimeout(monitoringTask, selectedSpeed.value);
            }
        };
        
        // 启动监控
        const startMonitoring = () => {
            if (modelReady.value && !isMonitoring.value) {
                isMonitoring.value = true;
                addLog('启动监控任务...', 'info');
                monitoringTask();
            } else if (!modelReady.value) {
                addLog('错误: 模型未初始化完成!', 'danger');
            }
        };
        
        // 停止监控
        const stopMonitoring = () => {
            if (isMonitoring.value) {
                isMonitoring.value = false;
                addLog('监控任务已停止', 'info');
            }
        };
        
        // 重置系统
        const resetSystem = () => {
            stopMonitoring();
            modelReady.value = false;
            deviceStatus.value = 'normal';
            trafficStatus.value = 'normal';
            processedDataPoints.value = 0;
            modelLoadProgress.value = 0;
            logs.value = [];
            memoryUsage.value = 0;
            accuracy.value = 92;
            anomalyDetectionRate.value = 88;
            
            if (trafficCtx) {
                trafficCtx.clearRect(0, 0, trafficCtx.canvas.width, trafficCtx.canvas.height);
            }
            
            if (deviceCtx) {
                deviceCtx.clearRect(0, 0, deviceCtx.canvas.width, deviceCtx.canvas.height);
            }
            
            // 释放模型
            if (trafficModel) {
                trafficModel.dispose();
                trafficModel = null;
            }
            
            if (imageModel) {
                imageModel.dispose();
                imageModel = null;
            }
            
            addLog('系统已重置，所有状态已清除!');
        };
        
        // 生命周期钩子
        onMounted(() => {
            const deviceCanvas = document.getElementById('deviceCanvas');
            const trafficCanvas = document.getElementById('trafficChart');
            
            if (deviceCanvas.getContext) {
                deviceCtx = deviceCanvas.getContext('2d');
                drawDeviceImage(deviceStatus.value);
            }
            
            if (trafficCanvas.getContext) {
                trafficCtx = trafficCanvas.getContext('2d');
                drawTrafficChart([], trafficStatus.value);
            }
            
            // 添加初始化日志
            addLog('系统初始化完成');
            addLog('请点击"初始化模型"按钮加载机器学习模型');
        });
        
        return {
            deviceStatus,
            trafficStatus,
            modelReady,
            isMonitoring,
            logs,
            memoryUsage,
            accuracy,
            anomalyDetectionRate,
            processedDataPoints,
            modelLoadProgress,
            speeds,
            selectedSpeed,
            deviceStatusDisplay,
            trafficStatusDisplay,
            initModels,
            startMonitoring,
            stopMonitoring,
            resetSystem
        };
    }
}).mount('#app');