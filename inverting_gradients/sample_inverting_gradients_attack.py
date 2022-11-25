try:
    # Note that in order to make breaching suit our attack scenario,
    # our ultimate task is to modify this library
    import breaching
except ModuleNotFoundError:
    print("python3 -m pip install breaching first")
    
import torch

# Redirects logs directly into the jupyter notebook
import logging, sys

def main():
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
    logger = logging.getLogger()

    cfg = breaching.get_config(overrides=[])
            
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    print("==========Setup==========")
    print(setup)

    # will make this part more automated
    cfg.case.data.partition="unique-class"
    cfg.case.user.user_idx = 24
    cfg.case.model='resnet18'
    cfg.case.data.name="CIFAR100"

    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)

    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)

    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

    # test
    # will figure out a way to plot the data properly
    user.plot(true_user_data)
    user.plot(reconstructed_user_data)

if __name__ == '__main__':
    main()