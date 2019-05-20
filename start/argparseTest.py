"""
argparse的用法，参考接口地址：https://docs.python.org/2/library/argparse.html#
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="start a argparse")

    parser.add_argument('--big', default=2, type=int)
    parser.add_argument('--output', required=True, help='try try')
    parser.add_argument('--input', required=True, help='try try')
    # 输入没有默认并且是必要的参数
    args = parser.parse_args(['--output', 'aa', '--input', 'bb'])
    process(args)


def process(args):
    print(args.big)
    a = args.output + '.walk'
    print(a)
    v = args.input
    print(v)


if __name__ == '__main__':
    main()
