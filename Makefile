# change remote GPU settings here
REMOTE_USER = .
REMOTE_HOST = .
REMOTE_DIR = /
# ----
LOCAL_DIR = .
RSYNC_FLAGS = -avzHP
TB_LOG_DIR = ./.tensorboard
EXCLUDE = --exclude ~/ \
		  --exclude .vscode/ \
		  --exclude **/*.pyc

REMOTE_EXCLUDE = --exclude .git/
LOCAL_EXCLUDE = --exclude .tensorboard/

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  sync     : synchronise bidirectionally"
	@echo "  toGPU    : synchronise local file to the host"
	@echo "  fromGPU  : synchronise remote file to local"
	@echo "  tb       : start tensorboard"
	@echo "  help     : show help info"

.PHONY: sync
sync:
	@echo "Running dry-run sync to remote..."
	rsync $(RSYNC_FLAGS) --dry-run $(EXCLUDE) $(LOCAL_DIR)/ $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)
	@echo "Running dry-run sync from remote..."
	rsync $(RSYNC_FLAGS) --dry-run $(EXCLUDE) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/ $(LOCAL_DIR)


.PHONY: toGPU
toGPU:
	@echo "Syncing to remote server..."
	rsync $(RSYNC_FLAGS) $(EXCLUDE) $(LOCAL_EXCLUDE) $(LOCAL_DIR)/ $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)


.PHONY: fromGPU
fromGPU:
	@echo "Syncing from remote server..."
	rsync $(RSYNC_FLAGS) $(EXCLUDE) $(REMOTE_EXCLUDE) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/ $(LOCAL_DIR)


.PHONY: tb
tb:
	@echo "Starting TensorBoard"
	tensorboard --logdir=$(TB_LOG_DIR)


.DEFAULT_GOAL := help