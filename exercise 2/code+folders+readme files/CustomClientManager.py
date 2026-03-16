#customclientmanager file
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional
import threading
import time

class CustomClientManager(ClientManager):
    def __init__(self):
        # Dictionary to hold clients: client_id -> ClientProxy
        self._clients: Dict[str, ClientProxy] = {}
        # Mutex lock to avoid race conditions when accessing _clients
        self._lock = threading.Lock()

    def num_available(self) -> int:
        with self._lock:
            return len(self._clients)

    def register(self, client: ClientProxy) -> bool:

        print(f"[ClientManager] Registered client: {client.cid}")
        with self._lock:
            if client.cid in self._clients:
                # Client already registered
                return False
            self._clients[client.cid] = client
            return True

    def unregister(self, client: ClientProxy) -> None:
        with self._lock:
            if client.cid in self._clients:
                del self._clients[client.cid]

    def all(self) -> Dict[str, ClientProxy]:
        with self._lock:
            # Return a shallow copy to prevent external modifications
            return dict(self._clients)

    def wait_for(self, num_clients: int, timeout: int) -> bool:
        """Waits until at least num_clients are registered or timeout (seconds) passes."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.num_available() >= num_clients:
                return True
            time.sleep(0.1)  # Sleep shortly to avoid busy-waiting
        return False

    def sample(
            
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[object] = None
    ) -> List[ClientProxy]:
        
        print(f"[ClientManager] Sampling {num_clients} clients (min required: {min_num_clients})")
        
        with self._lock:
            all_clients = list(self._clients.values())
        
        if min_num_clients is not None and len(all_clients) < min_num_clients:
            # Not enough clients to sample
            return []
        
        # For now, ignore criterion and just sample randomly (or take first n)
        # Could add criterion filtering later
        sampled_clients = all_clients[:num_clients]
        return sampled_clients
