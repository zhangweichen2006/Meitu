import argparse
import os
import time
import yaml
from sshtunnel import SSHTunnelForwarder
from paramiko.ssh_exception import PasswordRequiredException


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_tunnel(tunnel_cfg, use_ssh_pkey=True):
    ssh_key_value = tunnel_cfg.get('ssh_key')
    ssh_key_path = os.path.expanduser(ssh_key_value) if ssh_key_value else None

    forwarder_kwargs = dict(
        ssh_username=tunnel_cfg.get('ssh_user'),
        ssh_private_key_password=tunnel_cfg.get('ssh_key_pass'),
        ssh_password=tunnel_cfg.get('ssh_pass'),
        remote_bind_address=(tunnel_cfg['remote_host'], tunnel_cfg['remote_port']),
        local_bind_address=('0.0.0.0', tunnel_cfg['local_port']),
        allow_agent=True,
        look_for_keys=False,
        set_keepalive=60,
    )

    if use_ssh_pkey and ssh_key_path:
        forwarder_kwargs['ssh_pkey'] = ssh_key_path

    server = SSHTunnelForwarder(
        (tunnel_cfg['ssh_host'], tunnel_cfg.get('ssh_port', 22)),
        **forwarder_kwargs,
    )
    return server


def main():
    parser = argparse.ArgumentParser(description="支持多个 SSH 隧道的脚本（YAML 配置）")
    parser.add_argument('--config', required=True, help='配置文件路径')
    args = parser.parse_args()

    config = load_config(args.config)
    tunnels_cfg = config.get('tunnels', [])
    servers = []

    for tunnel in tunnels_cfg:
        print(f"🚀 启动隧道 [{tunnel['name']}]: localhost:{tunnel['local_port']} → {tunnel['remote_host']}:{tunnel['remote_port']}")

        agent_available = bool(os.environ.get('SSH_AUTH_SOCK'))
        key_pass_provided = bool(tunnel.get('ssh_key_pass'))
        prefer_agent = agent_available and not key_pass_provided

        try:
            # Prefer SSH agent when available to avoid decrypting local key file (silences key passphrase warnings)
            if prefer_agent:
                server = create_tunnel(tunnel, use_ssh_pkey=False)
            else:
                server = create_tunnel(tunnel, use_ssh_pkey=True)
            server.start()
        except PasswordRequiredException:
            # If key requires a passphrase and none provided, fall back to ssh-agent
            print(f"🔁 检测到密钥需要口令且未提供 `ssh_key_pass`，尝试使用 SSH Agent 重新建立: {tunnel.get('ssh_key')}")
            server = create_tunnel(tunnel, use_ssh_pkey=False)
            server.start()

        servers.append((tunnel['name'], server))

    try:
        print("✅ 所有隧道已建立，按 Ctrl+C 停止")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 捕获中断，正在关闭所有隧道...")
    finally:
        for name, server in servers:
            print(f"🔒 关闭隧道 [{name}]")
            server.stop()
        print("✅ 所有隧道已关闭")


if __name__ == '__main__':
    main()
