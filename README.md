# Setup Instructions (Multi-Machine Deployment)

This framework follows a distributed deployment architecture consisting of:

* 1 Server Machine
* Multiple Client Machines

---

## Step 1: Generate CKKS Context (Server Machine Only)

Run on the server machine:

```bash
python encryption/generate_context.py
```

This generates:

```text
ckks_server_context.bin
ckks_public_context.bin
```

Purpose:

* `ckks_server_context.bin` → contains secret key (server only)
* `ckks_public_context.bin` → shared with clients

---

## Step 2: Distribute Public Context to Clients

Copy:

```text
ckks_public_context.bin
```

from server machine to every client machine.

Do NOT copy:

```text
ckks_server_context.bin
```

to clients.

---

## Step 3: Prepare Local Data and Build FAISS Index (Client Machines)

Each client machine should maintain its own dataset.

Example:

```text
client/Client1/data/
client/Client2/data/
```

Build local vector indices:

```bash
python build_index.py
```

This creates:

```text
faiss_index/
├── index.faiss
├── chunks.npy
```

Each client builds its own FAISS index locally.

---

## Step 4: Configure Machine Roles

### Server Machine

Required files:

```text
server.py
server/
encryption/
config/
ckks_server_context.bin
ckks_public_context.bin
```

---

### Client Machines

Required files:

```text
client.py
client/
encryption/
models/
rag_pipeline/
ckks_public_context.bin
```

---

## Step 5: Start Server Machine

Run:

```bash
python server.py
```

The server:

* initializes global model
* loads CKKS secret context
* starts Flower server
* waits for client connections

---

## Step 6: Start Client Machines

Run on each client machine:

```bash
python client.py
```

Each client:

* loads local dataset
* loads FAISS index
* trains locally
* encrypts model weights
* sends encrypted updates

---

# Deployment Workflow

```text
Server Starts
      ↓
Clients Connect
      ↓
Receive Global Model
      ↓
Local Training
      ↓
Encrypt Weights
      ↓
Send Encrypted Updates
      ↓
Encrypted Aggregation
      ↓
Global Model Update
      ↓
Repeat
```

---

# Security Notes

Server Machine Stores:

```text
ckks_server_context.bin
ckks_public_context.bin
```

Client Machines Store:

```text
ckks_public_context.bin
```

Privacy guarantee:

```text
Clients encrypt weights
Server aggregates encrypted updates
Only aggregated model is decrypted
Raw client weights remain hidden
```
