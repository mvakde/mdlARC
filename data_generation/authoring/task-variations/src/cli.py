"""One launcher for task-variation stages and every dataset."""
import argparse,json,sys
from pathlib import Path
from .common.config import load_config,STAGES,PIPELINE,RESULTS
from .common.data import all_task_refs,load_puzzle
from .common.runner import run_task
from .common.storage import verify,atomic_write_json,update_index

def positive_int(value):
    number=int(value)
    if number < 1: raise argparse.ArgumentTypeError('must be at least 1')
    return number

def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=(*STAGES,'produce-programs','verify','check','review-ui','visualise'))
    parser.add_argument('action',nargs='?',default='run',choices=('run','dry-run','prepare','sample','list','inspect','serve','verify'))
    parser.add_argument('--dataset',choices=('arc2','arc1','arc1-filtered','conceptarc'),default='arc2')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--task','--puzzle-id',action='append',default=[])
    parser.add_argument('--split',choices=('auto','all','training','evaluation'),default='auto')
    parser.add_argument('--all',action='store_true',help='Select every task in the chosen dataset and split')
    parser.add_argument('--force',action='store_true',help='Replace selected output while preserving its existing version')
    parser.add_argument('--model');parser.add_argument('--effort')
    parser.add_argument('--count',type=int,default=42);parser.add_argument('--start-seed',type=int,default=0)
    parser.add_argument('--batch-size',type=positive_int,help='Select the first N reviewed families for produce-programs (independent of --count)')
    parser.add_argument('--workers',type=positive_int,default=1,help='Maximum concurrent produce-programs families (default: 1)')
    parser.add_argument('--resume',action='store_true',help='Skip verified program bundles and preserve/replace unsuccessful attempts')
    parser.add_argument('--num-train',type=int);parser.add_argument('--num-test',type=int)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--results-root',type=Path,default=RESULTS,help='Variation artifact root for produce-programs')
    parser.add_argument('--output-root',type=Path,help='Program-label artifact root')
    parser.add_argument('--dsl-root',type=Path,help='ARC DSL repository (default: ../arc-dsl)')
    parser.add_argument('--compiler',type=Path,help='Authored compiler.py exposing compile_pair(**request)')
    puzzle_source=parser.add_mutually_exclusive_group()
    puzzle_source.add_argument('--fresh',action='store_true',help='Generate fresh puzzles with v2 instead of reading saved seeds')
    puzzle_source.add_argument('--original',action='store_true',help='Label the original task once instead of variations; ignores --count and --start-seed')
    parser.add_argument('--timeout-seconds',type=int,default=60)
    parser.add_argument('--author-timeout-seconds',type=int,default=2700,help='Timeout for each Codex compiler-authoring turn (default: 2700 seconds / 45 minutes)')
    parser.add_argument('--author-puzzles',type=int,default=2,help='Maximum seed examples shown to the compiler author, independent of --count')
    parser.add_argument('--author-example-bytes',type=int,default=24000,help='Byte budget for solved author examples and repair counterexamples')
    parser.add_argument('--repair-turns',type=int,default=2,help='Codex repairs after failed compiler checks')
    parser.add_argument('--compiler-check-seeds',type=int,default=2,help='Additional fresh puzzles used to check the authored compiler')
    parser.add_argument('--reauthor-compiler',action='store_true',help='Create a replacement family compiler through Codex CLI')
    return parser

def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    if args.stage=='produce-programs':
        from .programs.pipeline import main as programs_main
        return programs_main(args,parser)
    if args.dataset=='arc1':
        parser.error('Full ARC1 selection requires produce-programs; reviewed sources are stored under arc2 and arc1-filtered')
    if args.original:
        parser.error('--original requires produce-programs')
    if args.batch_size is not None or args.workers != 1 or args.resume:
        parser.error('--batch-size, --workers and --resume require produce-programs')
    if args.action in ('inspect','serve','verify'):
        parser.error('This action requires produce-programs')
    if args.stage in ('verify','check'):
        report=verify(RESULTS);atomic_write_json(RESULTS/'checks/verification.json',report)
        print(json.dumps(report));return 0
    if args.stage=='visualise':
        import runpy
        sys.argv=[str(PIPELINE/'tools/visualiser/server.py'),'--source',str(RESULTS),'--port',str(args.port)]
        runpy.run_path(sys.argv[0],run_name='__main__');return 0
    if args.stage=='review-ui':
        import runpy
        sys.argv=[str(PIPELINE/'tools/manual-review/server.py'),'--config',str(args.config or PIPELINE/'configs'/(args.dataset+'.json')),'--port',str(args.port)]
        runpy.run_path(sys.argv[0],run_name='__main__');return 0
    config=load_config(args.config or PIPELINE/'configs'/(args.dataset+'.json'),args.stage)
    if args.model: config.default_model=args.model
    if args.effort: config.default_reasoning_effort=args.effort
    if args.all and args.task: parser.error('Select either --all or --task')
    if args.split=='all' and args.task: parser.error('Use --split auto with specific tasks')
    if args.task:
        targets=[load_puzzle(config,name,args.split) for name in args.task]
        refs=[(t.source_split,t.puzzle_id) for t in targets]
    elif args.all or args.action=='list':
        refs=[(t.source_split,t.puzzle_id) for t in all_task_refs(config,args.split)]
        if args.all and config.dataset_name=='arc2':
            pinned=(config.manual_summary_ids if config.stage=='summarise-task' else
                    config.example_ids if config.stage=='review-first-generation' else ())
            refs=[(s,n) for s,n in refs if not (s=='evaluation' and n in pinned)]
    else: parser.error('Select --task NAME or --all')
    errors=[]
    for split,name in refs:
        try:
            if args.action=='list': result={'task':name,'split':split}
            elif args.action=='sample':
                if args.stage not in ('generate-from-summary','generate-from-reviews'): parser.error('Sampling requires a generator stage')
                from .common.sampling import sample
                from .common.storage import current,task_lock
                task=load_puzzle(config,name,split).path.parent
                if not current(task,args.stage): raise ValueError('Selected generator needs updating')
                with task_lock(task):
                    m=sample(config,task,version=1 if args.stage=='generate-from-summary' else 2,
                        count=args.count,start_seed=args.start_seed,num_train=args.num_train,num_test=args.num_test)
                result={'task':name,'samples':str(m.parent)}
            else: result=run_task(config,name,split,args.force,args.action in ('dry-run','prepare'))
            print(json.dumps(result),flush=True)
        except Exception as exc:
            errors.append(name);print(json.dumps({'task':name,'error':str(exc)}),file=sys.stderr,flush=True)
    return int(bool(errors))
if __name__=='__main__': raise SystemExit(main())
