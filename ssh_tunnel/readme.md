# 🔐 SSH Tunnel Manager（多隧道配置版）

一个基于 Python 的命令行工具，支持通过 YAML 配置文件一次性建立多个基于 SSH 的本地端口转发隧道。适用于远程VNC、SSH、API等服务开发调试。

---

## ✅ 特性

- 支持 **多个 SSH 隧道并发连接**
- 支持 **公私钥验证**
- 支持本地端口 → 跳板机 → 内网服务转发
- 配置简单，适用于多服务或多环境场景

---

## 📦 安装依赖

```bash
pip install sshtunnel pyyaml
```

## 📝 配置文件示例
```yaml
tunnels:
  - name: ssh-tunnel
    ssh_host: jump1.example.com  // 跳板机地址
    ssh_port: 22  // 跳板机端口
    ssh_user: ubuntu  // 跳板机用户名
    ssh_pass:         // 跳板机密码， 可选
    ssh_key: ~/.ssh/id_rsa  // 跳板机私钥路径， 可选
    ssh_key_pass: null  // 跳板机私钥密码， 可选
    remote_host: 127.0.0.1  // 远程服务地址
    remote_port: 3306  // 远程服务端口
    local_port: 10022   // 本地端口, 自定义

  - name: vnc-tunnel
    ssh_host: jump1.example.com  // 跳板机地址
    ssh_port: 22  // 跳板机端口
    ssh_user: ubuntu  // 跳板机用户名
    ssh_pass:         // 跳板机密码， 可选
    ssh_key: ~/.ssh/id_rsa  // 跳板机私钥路径， 可选
    ssh_key_pass: null  // 跳板机私钥密码， 可选
    remote_host: 127.0.0.1  // 远程服务地址
    remote_port: 3306  // 远程服务端口
    local_port: 10022   // 本地端口, 自定义
```

## 🚀 运行
```
python ssh_tunnel_multi.py --config tunnels.yml
```