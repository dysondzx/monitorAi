<script setup>
import { ref, computed, onMounted, nextTick } from 'vue';
import * as tf from '@tensorflow/tfjs';

// 状态和数据
const deviceStatus = ref('normal');
const trafficStatus = ref('normal');
const modelReady = ref(false);
const isMonitoring = ref(false);
const logs = ref([]);
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

let deviceCtx = null;
let trafficCtx = null;

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

/**
 * 日志记录方法
 * @param {string} message - 日志消息
 * @param {string} type - 日志类型（info/warning/danger）
 */
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

/**
 * TensorFlow.js 模型初始化
 */
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
        console.error('详细的模型加载错误:', error);
        modelLoadProgress.value = 0;
    }
};

/**
 * 创建流量异常检测模型（纯前端实现）
 * @returns {Promise} - 模型实例
 */
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
        model.add(tf.layers.dense({ 
            units: 1,
            activation: 'sigmoid'
        }));
        
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

/**
 * 创建图像识别模型（纯前端实现）
 * @returns {Promise} - 模型实例
 */
const createImageModel = () => {
    return new Promise((resolve) => {
        addLog('创建设备状态识别模型...');
        
        // 创建一个简单的CNN模型，避免加载外部模型文件
        const createSimpleModel = async () => {
            try {
                const model = tf.sequential();
                
                // 卷积层
                model.add(tf.layers.conv2d({
                    inputShape: [224, 224, 3],
                    filters: 16,
                    kernelSize: 3,
                    activation: 'relu'
                }));
                model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
                model.add(tf.layers.conv2d({
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu'
                }));
                model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
                
                // 全连接层
                model.add(tf.layers.flatten());
                model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
                model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
                
                // 编译模型
                model.compile({
                    optimizer: tf.train.adam(0.001),
                    loss: 'categoricalCrossentropy',
                    metrics: ['accuracy']
                });
                
                // 模拟训练过程
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
            } catch (error) {
                addLog(`创建图像模型失败: ${error.message}`, 'danger');
                console.error('图像模型创建错误:', error);
                throw error;
            }
        };
        
        createSimpleModel().then(resolve);
    });
};

/**
 * 生成训练数据（纯前端模拟）
 * @returns {Object} - 包含inputs和labels的训练数据
 */
const generateTrainingData = () => {
    const inputs = [];
    const labels = [];
    
    for (let i = 0; i < 200; i++) {
        const seq = Array.from({ length: 10 }, () => Math.random() * 100);
        inputs.push(seq.map(v => [v]));
        
        // 简单规则：如果最后3个值中有1个大于80则视为异常
        const lastThree = seq.slice(-3);
        const isAnomaly = lastThree.some(v => v > 80) ? 1 : 0;
        labels.push([isAnomaly]);
    }
    
    return { inputs, labels };
};

/**
 * 生成设备模拟图像
 * @param {string} status - 设备状态
 */
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

/**
 * 绘制流量图表
 * @param {Array} data - 流量数据数组
 * @param {string} status - 流量状态
 */
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
    const warningThreshold = 80;
    const dangerThreshold = 90;
    
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

/**
 * 生成模拟流量数据
 * @returns {Array} - 流量数据数组
 */
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

/**
 * 使用模型进行流量异常检测
 * @param {Array} data - 流量数据
 * @returns {Promise<string>} - 检测结果状态
 */
const detectTrafficAnomaly = async (data) => {
    if (!trafficModel || !modelReady.value) return 'normal';
    
    try {
        // 确保数据有效性
        if (!Array.isArray(data) || data.length !== 10) {
            addLog(`流量数据格式不正确: 应为长度为10的数组`, 'warning');
            return 'normal';
        }
        
        // 确保所有数据都是有效数字
        if (data.some(v => typeof v !== 'number' || isNaN(v))) {
            addLog(`流量数据包含无效数字`, 'warning');
            return 'normal';
        }
        
        try {
            // 将数据转换为Tensor - 使用try-catch确保张量操作失败不会导致整个功能失效
            const normalizedData = data.map(v => [v / 100]); // 数据归一化
            const input = tf.tensor3d([normalizedData], [1, 10, 1]); // 明确指定形状
            
            // 进行预测
            const prediction = await trafficModel.predict(input);
            const value = prediction.dataSync()[0];

            // 清理内存
            tf.dispose([input, prediction]);
            
            // 限制预测值范围
            const clampedValue = Math.max(0, Math.min(1, value));
            
            // 根据预测值确定状态
            if (clampedValue > 0.6) {
                trafficStatus.value = 'danger';
                return 'danger';
            } else if (clampedValue > 0.4) {
                trafficStatus.value = 'warning';
                return 'warning';
            } else {
                trafficStatus.value = 'normal';
                return 'normal';
            }
        } catch (tensorError) {
            // 张量操作失败时，使用简单的启发式规则作为备用
            addLog(`张量操作失败: ${tensorError.message}，使用备用检测逻辑`, 'warning');
            
            // 启发式规则：如果数据中有任何值超过85，则视为危险；如果超过70，则视为警告
            const maxValue = Math.max(...data);
            if (maxValue > 85) {
                trafficStatus.value = 'danger';
                return 'danger';
            } else if (maxValue > 70) {
                trafficStatus.value = 'warning';
                return 'warning';
            } else {
                trafficStatus.value = 'normal';
                return 'normal';
            }
        }
    } catch (error) {
        addLog(`流量检测失败: ${error.message}`, 'warning');
        return 'normal';
    }
};

/**
 * 图像识别处理流程
 * @returns {Promise<string>} - 设备状态识别结果
 */
const processDeviceImage = async () => {
    if (!imageModel || !deviceCtx || !modelReady.value) return 'normal';
    
    try {
        const canvas = deviceCtx.canvas;
        
        // 从Canvas获取图像数据
        const imageData = deviceCtx.getImageData(0, 0, canvas.width, canvas.height);
        
        try {
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
        } catch (tensorError) {
            // 对于张量操作错误，提供备用逻辑
            addLog(`张量操作失败: ${tensorError.message}，使用备用检测逻辑`, 'warning');
            
            // 根据当前设备状态简单模拟下一个状态的转移
            // 这是一个备用方案，确保系统可以继续运行
            const currentStatus = deviceStatus.value;
            let nextStatus = currentStatus;
            
            // 随机决定是否改变状态
            if (Math.random() < 0.1) {
                const possibleStatuses = ['normal', 'warning', 'danger'];
                // 移除当前状态，从剩余的中随机选择
                const otherStatuses = possibleStatuses.filter(s => s !== currentStatus);
                nextStatus = otherStatuses[Math.floor(Math.random() * otherStatuses.length)];
            }
            
            deviceStatus.value = nextStatus;
            return nextStatus;
        }
    } catch (error) {
        addLog(`图像识别失败: ${error.message}`, 'warning');
        return 'normal';
    }
};

/**
 * 监控任务主循环
 */
const monitoringTask = async () => {
    if (!isMonitoring.value) return;
    
    // 任务1: 设备状态识别
    drawDeviceImage(deviceStatus.value);
    const imgStatus = await processDeviceImage();
    
    // 任务2: 流量异常检测
    const trafficData = generateTrafficData();
    const trafficStatusResult = await detectTrafficAnomaly(trafficData);
    drawTrafficChart(trafficData, trafficStatusResult);
    
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
        
        const msgs = trafficStatusResult === 'normal' ? normalMsgs : 
                    trafficStatusResult === 'warning' ? warningMsgs : dangerMsgs;
        
        const msgType = trafficStatusResult === 'normal' ? 'info' : 
                        trafficStatusResult === 'warning' ? 'warning' : 'danger';
        
        const randomMsg = msgs[Math.floor(Math.random() * msgs.length)];
        addLog(randomMsg, msgType);
    }
    
    // 下一个循环
    if (isMonitoring.value) {
        setTimeout(monitoringTask, selectedSpeed.value);
    }
};

/**
 * 启动监控
 */
const startMonitoring = () => {
    if (modelReady.value && !isMonitoring.value) {
        isMonitoring.value = true;
        addLog('启动监控任务...', 'info');
        monitoringTask();
    } else if (!modelReady.value) {
        addLog('错误: 模型未初始化完成!', 'danger');
    }
};

/**
 * 停止监控
 */
const stopMonitoring = () => {
    if (isMonitoring.value) {
        isMonitoring.value = false;
        addLog('监控任务已停止', 'info');
    }
};

/**
 * 重置系统
 */
const resetSystem = () => {
    stopMonitoring();
    deviceStatus.value = 'normal';
    trafficStatus.value = 'normal';
    modelReady.value = false;
    logs.value = [];
    modelLoadProgress.value = 0;
    selectedSpeed.value = 2000;
    
    // 释放模型资源
    if (trafficModel) {
        trafficModel.dispose();
        trafficModel = null;
    }
    if (imageModel) {
        imageModel.dispose();
        imageModel = null;
    }
    
    // 清理画布
    if (deviceCtx) {
        deviceCtx.clearRect(0, 0, deviceCtx.canvas.width, deviceCtx.canvas.height);
    }
    if (trafficCtx) {
        trafficCtx.clearRect(0, 0, trafficCtx.canvas.width, trafficCtx.canvas.height);
    }
    
    addLog('系统已重置', 'info');
};

// 组件挂载后初始化Canvas引用
onMounted(async () => {
    await nextTick();
    
    // 获取Canvas上下文
    const deviceCanvas = document.getElementById('deviceCanvas');
    const trafficChart = document.getElementById('trafficChart');
    
    if (deviceCanvas) {
        deviceCtx = deviceCanvas.getContext('2d');
    }
    
    if (trafficChart) {
        trafficCtx = trafficChart.getContext('2d');
    }
    
    // 等待tf.js加载完成
    if (typeof tf !== 'undefined') {
        addLog('TensorFlow.js 已加载完成，可以开始使用', 'info');
    } else {
        addLog('TensorFlow.js 未加载，请检查依赖', 'danger');
    }
});
</script>

<template>
    <div class="app-container">
        <h1>前端智能监控原型系统</h1>
        <p class="simulation-speed">
            <strong>模拟速度调节：</strong>
            <label v-for="(speed, index) in speeds" :key="index">
                <input type="radio" v-model="selectedSpeed" :value="speed.value"> {{ speed.label }}
            </label>
        </p>

        <div class="model-controls">
            <button class="btn btn-primary" @click="initModels">初始化模型</button>
            <button class="btn btn-success" @click="startMonitoring" :disabled="!modelReady">启动监控</button>
            <button class="btn btn-danger" @click="stopMonitoring">停止监控</button>
            <button class="btn" @click="resetSystem">重置系统</button>
        </div>

        <div class="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" :style="{ width: modelLoadProgress + '%' }"></div>
            </div>
            <div v-if="modelLoadProgress < 100">模型加载中: {{ modelLoadProgress }}%</div>
            <div v-else style="color:#27ae60">模型已准备就绪!</div>
        </div>

        <div class="dashboard">
            <div class="card">
                <h2 class="section-title">设备状态监控</h2>
                <div class="status-display">
                    <p v-if="modelReady">
                        <span :class="'status-indicator status-' + deviceStatus"></span>
                        当前状态: <strong>{{ deviceStatusDisplay }}</strong>
                    </p>
                    <p v-else>模型未启动</p>
                </div>
                <canvas id="deviceCanvas" width="500" height="300"></canvas>
            </div>

            <div class="card">
                <h2 class="section-title">网络流量异常检测</h2>
                <div>
                    <p v-if="modelReady">
                        <span :class="'status-indicator status-' + trafficStatus"></span>
                        网络状态: <strong>{{ trafficStatusDisplay }}</strong>
                    </p>
                    <p v-else>模型未启动</p>
                </div>
                <canvas id="trafficChart" width="500" height="300"></canvas>
            </div>
        </div>

        <div class="card">
            <h2 class="section-title">系统日志</h2>
            <div class="log-container">
                <div v-for="(log, index) in logs" :key="index" class="log-entry" :class="log.type">
                    {{ log.message }}
                </div>
            </div>
        </div>
    </div>
</template>



