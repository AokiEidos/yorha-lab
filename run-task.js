#!/usr/bin/env node

/**
 * YoRHa::LaB 任务执行器
 * 用于 Cron 定时触发或手动执行
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TASKS_FILE = path.join(__dirname, 'tasks.json');
const LOG_DIR = path.join(__dirname, 'log');

// 确保日志目录存在
if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
}

function loadTasks() {
  const data = fs.readFileSync(TASKS_FILE, 'utf-8');
  return JSON.parse(data);
}

function saveTasks(tasks) {
  fs.writeFileSync(TASKS_FILE, JSON.stringify(tasks, null, 2));
}

function getNextPendingTask(tasks) {
  // 排除在 exclude_hours 时段的任务
  const now = new Date();
  const hour = now.getHours();
  const excludeHours = tasks.execution_rules?.exclude_hours || [];

  for (const task of tasks.queue) {
    if (task.status === 'pending' && !excludeHours.includes(hour)) {
      return task;
    }
  }
  return null;
}

function log(taskId, message) {
  const timestamp = new Date().toISOString();
  const logFile = path.join(LOG_DIR, `${taskId}.log`);
  const logEntry = `[${timestamp}] ${message}\n`;
  fs.appendFileSync(logFile, logEntry);
  console.log(logEntry.trim());
}

async function executeTask(task) {
  log(task.id, `开始执行任务: ${task.task}`);

  // 更新任务状态为 in_progress
  task.status = 'in_progress';

  const tasks = loadTasks();
  const taskIndex = tasks.queue.findIndex((t) => t.id === task.id);
  if (taskIndex !== -1) {
    tasks.queue[taskIndex] = task;
    saveTasks(tasks);
  }

  log(task.id, `任务标记为 in_progress，请手动完成或扩展执行逻辑`);

  return task;
}

function main() {
  console.log('🔧 YoRHa::LaB 任务执行器\n');

  const tasks = loadTasks();
  const nextTask = getNextPendingTask(tasks);

  if (!nextTask) {
    console.log('✅ 没有待执行的 pending 任务');
    process.exit(0);
  }

  console.log(`🎯 下一个任务: [Phase] ${nextTask.task}`);

  executeTask(nextTask).then((task) => {
    console.log(`\n✅ 任务 ${task.id} 已开始执行`);
    console.log(`   请查看日志: ~/workspace/yorha-lab/log/${task.id}.log`);
  });
}

main();
