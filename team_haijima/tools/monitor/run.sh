docker run --rm -t --name monitor \
-v $HOME/workspaces/kira/.devcontainer/denv/gcloud:/root/.config/gcloud \
-v $HOME/workspaces/kira/.devcontainer/denv/.ssh:/root/.ssh \
geniac_haijima/monitor:latest
