# トラブルシュート

- 様々な理由で学習が止まることがあります。その際の原因の発見方法についての記載です。
- 監視ツール等で学習が止まっていることを検出した場合には、以下の方法で原因を確認してください。

## JOBが実行されていないことを確認

- 期待する結果：リストに表示されない。
- リストに残っている場合、処理が実行されているにも関わらず、学習が止まったなどの判定が行われている。
  - 本ケースはまだ遭遇していないが、ケースとして考慮が必要
  - 発生の時に考える

```bash
squeue
```

## JOB番号が分かっている場合

- 期待する結果：JOBの詳細が表示される。
  - ステータスがFAILED：プロセスが失敗
  - ステータスがCANCELLED：scancel等でキャンセルされた場合（人為的）
  - ステータスがCOMPLETED：プロセスが正常終了した場合（学習であり得るのか？最後まで学習が回ったなど？）
- JOBの詳細が表示されない場合には、JOB番号が間違っている可能性がある。

```bash
sacct --jobs <JOB番号>
```

## sbatchのログ確認

- `/storage3/jobs_outs/` 配下のログを確認する。
  - wandbにもログを確認する方法があるが、300行程度しか確認できない模様。
    - ただし、ノード別に分かれてログが管理される（+ユニークログ？重要そうなログのみに集約される？）ので、見方によっては有効な場合もある。（sbatchのログは全てのノードがまとまって出力される）
  - 学習時のログは初期の学習段階でも数千、数万行を超えるので状況に応じて
- ただし、基本的にはファイルサイズが大きいので以下のコマンドで確認をする。（その他、moreやviなどがあるが、lessがファイルを開くのに一番早かった）

```bash
less /storage3/jobs_outs/pretrain_<学習しているモデル>_<JOB番号>.out
```

### lessでエラーを探す

- `less`でファイルを開いたら、`/Error`と入力しエンターを押す。
  - Errorの文字が含まれる行がヒットするはずなので、カーソルで上下を行い例外やエラーの状況を確認する。
  - `n`キーで次のError文へ、`Shift + n`で前のError文へ移動する。
  - マルチノードだとエラーの発生順にログが書かれる訳ではないので、必ず複数回検索を行い、可能な限り（※）全てのエラーを確認すること。
  - ※パラメータ次第ではエラーの数も多いため、全てのエラーを確認するのは難しい場合もある。
- `/Error`で見つからなかった場合には、`/Exception`で同様にファイルを確認する。（今後もエラーとなるキーワードは追加される可能性がある。）

## よくあるエラー

- NCCL通信エラー
  - 何かしらのノードのエラーが発生した際に起きるエラー。ここの箇所に詳細は書かれていないため、他のエラーを探す必要がある。
  - （例えばマスタポートを既に使っている等のエラーなどで別のノードでの起動に失敗した時など）
  - ただし、ABEJA様も苦労したというノード自体の障害も疑う必要もあるエラー。他のエラーの原因がなければ可能性はあるが、モデル固有で発生する場合にはモデルを疑うべきかと思われる。

```
slurm0-a3-ghpc-8: [rank17]:[E ProcessGroupNCCL.cpp:523] [Rank 2] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=4, OpType=ALLREDUCE, NumelIn=9437184, NumelOut=9437184, Timeout(ms)=600000) ran for 600070 milliseconds before timing out.
slurm0-a3-ghpc-8: [rank17]:[E ProcessGroupNCCL.cpp:537] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
slurm0-a3-ghpc-8: [rank17]:[E ProcessGroupNCCL.cpp:543] To avoid data inconsistency, we are taking the entire process down.
slurm0-a3-ghpc-8: [rank17]:[E ProcessGroupNCCL.cpp:1182] [Rank 2] NCCL watchdog thread terminated with exception: [Rank 2] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=4, OpType=ALLREDUCE, NumelIn=9437184, NumelOut=9437184, Timeout(ms)=600000) ran for 600070 milliseconds before timing out.
slurm0-a3-ghpc-8: Exception raised from checkTimeout at /opt/conda/conda-bld/pytorch_1704987288773/work/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:525 (most recent call first):
slurm0-a3-ghpc-8: frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x57 (0x154ce8785d87 in /storage3/miniconda3/envs/.cuda12/lib/python3.11/site-packages/torch/lib/libc10.so)
slurm0-a3-ghpc-8: frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x1e6 (0x154ce994a4d6 in /storage3/miniconda3/envs/.cuda12/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
slurm0-a3-ghpc-8: frame #2: c10d::ProcessGroupNCCL::workCleanupLoop() + 0x19d (0x154ce994da2d in /storage3/miniconda3/envs/.cuda12/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
slurm0-a3-ghpc-8: frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x119 (0x154ce994e629 in /storage3/miniconda3/envs/.cuda12/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so)
slurm0-a3-ghpc-8: frame #4: <unknown function> + 0xdbbf4 (0x154d3521cbf4 in /storage3/miniconda3/envs/.cuda12/lib/python3.11/site-packages/torch/lib/../../../.././libstdc++.so.6)
slurm0-a3-ghpc-8: frame #5: <unknown function> + 0x8609 (0x154d3d976609 in /lib/x86_64-linux-gnu/libpthread.so.0)
slurm0-a3-ghpc-8: frame #6: clone + 0x43 (0x154d3d741353 in /lib/x86_64-linux-gnu/libc.so.6)
```

- CUDA out of memory
  - VRAMにモデルが乗らない時に発生するエラー。現時点ではモデルは調整済みではあるものの警戒は必要かと思われる。
  - 初期段階では良いが、学習後半で発生するようになった場合は、どう対応するべきか？

- その他、発生したエラーの詳細についてはNotionに纏まっているので、そちらを参照すること。
  
### ゾンビプロセスの確認

- 学習が止まった際に、プロセスが残っている場合がある。その際には以下のコマンドで確認を行う。

```bash
ssh slurm0-a3-ghpc-6
nvidia-smi
```

- squeueでジョブが無いにも関わらず、プロセスが確認できることがある。`925944`はゾンビプロセスとなる。

```
Sun Apr 28 22:15:57 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 80GB HBM3          On  | 00000000:04:00.0 Off |                    0 |
| N/A   38C    P0             113W / 700W |    533MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  | 00000000:05:00.0 Off |                    0 |
| N/A   34C    P0              67W / 700W |      7MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  | 00000000:0A:00.0 Off |                    0 |
| N/A   36C    P0              69W / 700W |      7MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  | 00000000:0B:00.0 Off |                    0 |
| N/A   33C    P0              68W / 700W |      7MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   4  NVIDIA H100 80GB HBM3          On  | 00000000:84:00.0 Off |                    0 |
| N/A   36C    P0              71W / 700W |      7MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   5  NVIDIA H100 80GB HBM3          On  | 00000000:85:00.0 Off |                    0 |
| N/A   44C    P0              69W / 700W |      7MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   6  NVIDIA H100 80GB HBM3          On  | 00000000:8A:00.0 Off |                    0 |
| N/A   34C    P0              68W / 700W |      7MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   7  NVIDIA H100 80GB HBM3          On  | 00000000:8B:00.0 Off |                    0 |
| N/A   33C    P0              68W / 700W |      7MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A    925944      C   python                                      520MiB |
+---------------------------------------------------------------------------------------+
```

- ゾンビプロセスの削除方法については[こちらを参照](./zombie.md)。
