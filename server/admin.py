"""
Operator management CLI for the TRELLIS.2 web app.

This is how the lab admin manages the login allow-list. Run it from the repo
root inside the venv:

    python -m server.admin add    a.hulaimi         # prompts for a password
    python -m server.admin add    a.hulaimi --password 's3cret'
    python -m server.admin list
    python -m server.admin passwd  a.hulaimi         # change an existing password
    python -m server.admin remove  j.haddad

IMPORTANT: adding the *first* operator is what turns authentication ON. While
the operators table is empty the server runs open (bootstrap mode) so you can
never lock yourself out of the machine; the moment one operator exists, every
/api/* route requires a valid login. After changing operators on the launchd
box, restart it:

    launchctl kickstart -k gui/$(id -u)/com.trellis.webserver
"""

import argparse
import getpass
import sys

from . import db


def _prompt_password() -> str:
    pw = getpass.getpass("New password: ")
    if not pw:
        print("Password must not be empty.", file=sys.stderr)
        sys.exit(1)
    if getpass.getpass("Confirm password: ") != pw:
        print("Passwords did not match.", file=sys.stderr)
        sys.exit(1)
    return pw


def cmd_add(args: argparse.Namespace) -> None:
    password = args.password or _prompt_password()
    existed = any(o["username"] == args.username.strip() for o in db.list_operators())
    first = db.count_operators() == 0
    db.add_operator(args.username, password)
    verb = "Updated" if existed else "Added"
    print(f"{verb} operator '{args.username.strip()}'.")
    if first:
        print(
            "\n  This is the FIRST operator -- authentication is now ON.\n"
            "  Restart the server so it picks up the change:\n"
            "    launchctl kickstart -k gui/$(id -u)/com.trellis.webserver"
        )


def cmd_passwd(args: argparse.Namespace) -> None:
    if not any(o["username"] == args.username.strip() for o in db.list_operators()):
        print(f"No such operator: '{args.username.strip()}'.", file=sys.stderr)
        sys.exit(1)
    password = args.password or _prompt_password()
    db.add_operator(args.username, password)  # upsert = password change
    print(f"Password updated for '{args.username.strip()}'.")


def cmd_remove(args: argparse.Namespace) -> None:
    if db.remove_operator(args.username):
        print(f"Removed operator '{args.username.strip()}'.")
        if db.count_operators() == 0:
            print(
                "\n  No operators remain -- authentication is now OFF (open bootstrap mode).\n"
                "  Add one again before exposing this server beyond your LAN."
            )
    else:
        print(f"No such operator: '{args.username.strip()}'.", file=sys.stderr)
        sys.exit(1)


def cmd_list(_args: argparse.Namespace) -> None:
    ops = db.list_operators()
    if not ops:
        print("(no operators -- auth is OFF; the app is open until you add one)")
        return
    print(f"{len(ops)} operator(s):")
    for o in ops:
        flag = "  [disabled]" if o["disabled"] else ""
        print(f"  - {o['username']}{flag}")


def main(argv: list[str] | None = None) -> None:
    db.init_db()
    parser = argparse.ArgumentParser(prog="python -m server.admin", description="Manage login operators.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add (or reset) an operator")
    p_add.add_argument("username")
    p_add.add_argument("--password", help="Set non-interactively (otherwise prompted)")
    p_add.set_defaults(func=cmd_add)

    p_pw = sub.add_parser("passwd", help="Change an existing operator's password")
    p_pw.add_argument("username")
    p_pw.add_argument("--password", help="Set non-interactively (otherwise prompted)")
    p_pw.set_defaults(func=cmd_passwd)

    p_rm = sub.add_parser("remove", help="Remove an operator")
    p_rm.add_argument("username")
    p_rm.set_defaults(func=cmd_remove)

    p_ls = sub.add_parser("list", help="List operators")
    p_ls.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
