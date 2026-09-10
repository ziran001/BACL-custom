"""Default to original BACL; opt in explicitly to the legacy TorchVision port."""
from bacl_official.dispatch import dispatch


def main():
    dispatch('evaluate')


if __name__ == '__main__':
    main()
